"""Run pipeline: which stages of a run can start, dispatch to workers, follow jobs, list their outputs."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
import httpx
from dataengine.variation import load_cells
from orchestrator.stages import STAGE_BY_ID, STAGES, StageDef, stage_argv
from orchestrator.store import Job, Run, Store

ACTIVE = ("queued", "running")
MAX_FILES = 200


class PipelineError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def worker_headers() -> dict[str, str]:
    token = os.environ.get("CATHLAB_WORKER_TOKEN")
    return {"X-Cathlab-Token": token} if token else {}


def worker_request(method: str, url: str, timeout: float = 5.0, **kw) -> httpx.Response:
    return httpx.request(method, url, headers=worker_headers(), timeout=timeout, **kw)


class Pipeline:
    def __init__(self, store: Store, var_dir: Path, repo_root: Path, workers_fn: Callable[[], list],
                 engines_fn: Callable[[], dict], http: Callable[..., httpx.Response] = worker_request,
                 resolve_policy: Callable[[str], str] | None = None,
                 local_actions: dict[str, Callable[[Run, dict], dict]] | None = None):
        self.store, self.var_dir, self.repo_root = store, Path(var_dir), Path(repo_root)
        self.local_actions = local_actions or {}
        self._workers_fn, self._engines_fn, self._http = workers_fn, engines_fn, http
        self._start_lock = threading.Lock()
        self._resolve_policy = resolve_policy

    # workers ---------------------------------------------------------------------------------
    def health(self, entry) -> dict:
        try:
            r = self._http("GET", entry.url.rstrip("/") + "/health", timeout=0.5)
            r.raise_for_status()
            return {"name": entry.name, "url": entry.url, "healthy": True, **r.json(), "error": ""}
        except Exception as e:  # noqa: BLE001 - any failure means "offline"
            return {"name": entry.name, "url": entry.url, "healthy": False, "capabilities": [], "error": str(e)[:200]}

    def workers(self) -> list[dict]:
        entries = self._workers_fn()
        with ThreadPoolExecutor(max_workers=max(1, len(entries))) as pool:
            return list(pool.map(self.health, entries))

    # stages ----------------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.var_dir / "runs" / run_id

    def latest(self, run_id: str, stage_id: str) -> Job | None:
        jobs = [j for j in self.store.list_jobs(kind="stage", run_id=run_id) if j.name == stage_id]
        return max(jobs, key=lambda j: j.created, default=None)

    def _check(self, run: Run, stage: StageDef, engines: dict, workers: list) -> tuple:
        """(reason the stage can't start or None, the worker it would run on or None)."""
        nr = run.selection.get("next_run")
        if stage.id == "generate" and nr:
            try:
                load_cells(nr)
            except (OSError, ValueError, KeyError):
                return f"next run file missing: {nr}", None
        if stage.local:
            return self._needs(run, stage), None
        engine_id = run.selection.get("simulator")
        manifest = engines.get(engine_id)
        if manifest is None:
            return f"no engine plugin '{engine_id}' (engines/{engine_id}/engine.yaml)", None
        if not stage.module:
            cap = stage.capability_for(run.selection)
            status = manifest.status(cap)
            if status != "ready":
                return f"{engine_id}: '{cap}' is {status.replace('_', ' ')}", None
        if stage.worker_capability:
            worker = next((w for w in workers if w["healthy"] and stage.worker_capability in w.get("capabilities", [])),
                          None)
            if worker is None:
                return f"no online worker with capability '{stage.worker_capability}'", None
        else:
            worker = next((w for w in workers if w["name"] == manifest.worker), None)
            if worker is None or not worker["healthy"]:
                return f"worker '{manifest.worker}' is offline", None
        return self._needs(run, stage), worker

    def _needs(self, run: Run, stage: StageDef) -> str | None:
        for need in stage.needs:
            job = self.latest(run.id, need)
            if job is None or job.state != "done":
                return f"needs '{need}' to finish first"
        return None

    def run_defaults(self, run: Run, stage: StageDef) -> dict:
        """Param defaults that come from the run itself: a next run's generate targets its failing cells."""
        nr = run.selection.get("next_run")
        if stage.id != "generate" or not nr:
            return {}
        try:
            return {"split": f"cells:{nr}", "start": 0, "n": sum(int(c.get("k", 8)) for c in load_cells(nr))}
        except (OSError, ValueError, KeyError):
            return {}

    def effective_params(self, run: Run, stage: StageDef, params: dict) -> dict:
        """User params over run defaults; `registry:<name>@<version|production>` resolved to an engine policy."""
        out = {**self.run_defaults(run, stage), **{k: v for k, v in params.items() if v is not None and v != ""}}
        policy = out.get("policy")
        if isinstance(policy, str) and policy.startswith("registry:"):
            ref = policy[len("registry:"):]
            if self._resolve_policy is None:
                raise PipelineError("no model registry configured", 400)
            try:
                out["policy"] = self._resolve_policy(ref)
            except (KeyError, ValueError) as e:
                raise PipelineError(f"cannot resolve {policy}: {e}", 400) from e
            if stage.id == "eval" and not out.get("model-version"):
                out["model-version"] = ref
        return out

    def stages(self, run: Run) -> list[dict]:
        engines, workers = self._engines_fn(), self.workers()
        out = []
        for stage in STAGES:
            reason, worker = self._check(run, stage, engines, workers)
            job = self.latest(run.id, stage.id)
            if job is not None and job.state in ACTIVE:
                job = self.refresh(job)
            running = job is not None and job.state in ACTIVE
            defaults = self.run_defaults(run, stage)
            params = [{**p, "default": defaults.get(p["name"], p["default"])} for p in stage.param_dicts()]
            out.append({"id": stage.id, "label": stage.label, "params": params,
                        "enabled": reason is None and not running, "reason": reason,
                        "worker": worker["name"] if worker else None, "job": job.model_dump() if job else None})
        return out

    def start(self, run: Run, stage_id: str, params: dict) -> Job:
        stage = STAGE_BY_ID.get(stage_id)
        if stage is None:
            raise PipelineError(f"unknown stage '{stage_id}'", 404)
        engines = self._engines_fn()
        reason, worker = self._check(run, stage, engines, self.workers())
        if reason:
            raise PipelineError(reason)
        try:
            flags = stage_argv(stage, self.effective_params(run, stage, params))
        except ValueError as e:
            raise PipelineError(str(e), 400) from e
        if stage.local:
            return self._run_local(run, stage, self.effective_params(run, stage, params))
        manifest = engines[run.selection["simulator"]]
        with self._start_lock:
            latest = self.latest(run.id, stage_id)
            if latest is not None and latest.state in ACTIVE:
                raise PipelineError(f"'{stage_id}' is already running ({latest.id})")
            job = self.store.create_job("stage", stage.id, params, run_id=run.id, worker=worker["name"])
        rd = self.run_dir(run.id)
        out = rd / stage.id / job.id
        exe = worker.get("executable") or "python"
        if stage.module:
            argv = [exe, "-m", stage.module, *stage.module_args, "--out", str(out), *flags]
        else:
            argv = [exe, "-m", manifest.entry, stage.cli, "--run", str(rd / "run.yaml"), "--out", str(out), *flags]
        for flag, source in stage.inputs.items():
            argv += [f"--{flag}", self.latest(run.id, source).result["out"]]
        try:
            r = self._http("POST", worker["url"].rstrip("/") + "/jobs",
                           json={"argv": argv, "name": f"{run.id}:{stage.id}", "cwd": str(self.repo_root)})
            r.raise_for_status()
            worker_job_id = r.json()["id"]
        except Exception as e:  # noqa: BLE001
            self.store.update_job(job.id, state="failed",
                                  result={"error": f"dispatch failed: {e}"[:300], "argv": argv, "out": str(out)})
            raise PipelineError(f"could not start on worker '{worker['name']}': {e}", 502) from e
        return self.store.update_job(job.id, state="running", result={
            "worker_job": worker_job_id, "worker_url": worker["url"], "argv": argv, "out": str(out)})


    def _run_local(self, run: Run, stage: StageDef, params: dict) -> Job:
        action = self.local_actions.get(stage.id)
        if action is None:
            raise PipelineError(f"stage '{stage.id}' is not available here", 501)
        with self._start_lock:
            job = self.store.create_job("stage", stage.id, params, run_id=run.id, worker=None)
        try:
            result = action(run, params)
        except (ValueError, KeyError, OSError) as e:
            msg = str(e).strip("'\"")
            return self.store.update_job(job.id, state="failed", result={"error": msg[:500]})
        if result.get("error"):
            return self.store.update_job(job.id, state="failed", result=result)
        return self.store.update_job(job.id, state="done", result=result)

    # jobs ------------------------------------------------------------------------------------
    def _worker_url(self, job: Job, suffix: str = "") -> str:
        return f"{job.result['worker_url'].rstrip('/')}/jobs/{job.result['worker_job']}{suffix}"

    def refresh(self, job: Job) -> Job:
        if job.state not in ACTIVE or "worker_job" not in job.result:
            return job
        try:
            r = self._http("GET", self._worker_url(job))
        except Exception:  # noqa: BLE001 - worker unreachable: keep the last known state
            return job
        if r.status_code == 404:
            return self.store.update_job(job.id, state="failed", result={"error": "the worker lost the job (restarted?)"})
        if r.status_code != 200:
            return job
        state = r.json()
        if state["state"] != job.state:
            job = self.store.update_job(job.id, state=state["state"], result={"returncode": state.get("returncode")})
        return job

    def log(self, job: Job, offset: int = 0) -> dict:
        job = self.refresh(job)
        if "worker_job" not in job.result:
            text = job.result.get("error") or job.result.get("message", "")
            return {"text": text[offset:], "offset": len(text), "state": job.state}
        try:
            r = self._http("GET", self._worker_url(job, "/log"), params={"offset": offset})
            r.raise_for_status()
            data = r.json()
            return {"text": data["text"], "offset": data["offset"], "state": job.state}
        except Exception as e:  # noqa: BLE001
            return {"text": "", "offset": offset, "state": job.state, "error": str(e)[:200]}

    def cancel(self, job: Job) -> Job:
        if job.state in ACTIVE and "worker_job" in job.result:
            try:
                self._http("POST", self._worker_url(job, "/cancel"))
            except Exception as e:  # noqa: BLE001
                raise PipelineError(f"cancel failed: {e}", 502) from e
        return self.refresh(job)

    def _out(self, job: Job) -> Path:
        if "out" not in job.result:
            raise PipelineError("job has no outputs", 404)
        return Path(job.result["out"]).resolve()

    def files(self, job: Job) -> list[dict]:
        base = self._out(job)
        if not base.exists():
            return []
        files = sorted(p for p in base.rglob("*") if p.is_file())[:MAX_FILES]
        return [{"path": str(p.relative_to(base)), "size": p.stat().st_size} for p in files]

    def file(self, job: Job, rel: str) -> Path:
        base = self._out(job)
        path = (base / rel).resolve()
        if base not in path.parents:
            raise PipelineError("path is outside the job's outputs", 400)
        if not path.is_file():
            raise PipelineError(f"no file '{rel}'", 404)
        return path
