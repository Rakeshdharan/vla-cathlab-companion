"""Harness job lifecycle: worktree -> agent -> path guard -> acceptance checks -> commit -> review -> push/discard.

State order: created, worktree, agent_running, guard, checks, ready_for_review, then pushed | discarded,
or failed (with `error`) at any step.
"""
import os
import threading
import time
import uuid
from fnmatch import fnmatch
from pathlib import Path
from pydantic import BaseModel, Field
from harness import gitops
from harness.agents import AgentsConfig, build_argv
from harness.guard import path_violations
from harness.model import TaskTemplate, allowed_for, checks_for, render_prompt, validate_params
from orchestrator.process import read_log, start_process


class HarnessSettings(BaseModel):
    repo: Path
    var_dir: Path                 # job records + logs: <var_dir>/agent/<id>/
    worktrees_dir: Path           # job worktrees: <worktrees_dir>/<id>/
    base_ref: str = "origin/main"
    remote: str = "origin"
    fetch: bool = True


class CheckResult(BaseModel):
    name: str
    ok: bool
    state: str


class AgentJob(BaseModel):
    id: str
    task: str
    title: str
    params: dict
    extra: str = ""
    agent: str
    branch: str
    worktree: str
    state: str = "created"
    error: str = ""
    base_sha: str = ""
    head_sha: str = ""
    changed_files: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    created: float
    updated: float


class HarnessError(RuntimeError):
    pass


class Harness:
    def __init__(self, settings: HarnessSettings, templates: dict[str, TaskTemplate], agents: AgentsConfig):
        self.s, self.templates, self.agents = settings, templates, agents
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    # persistence ---------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.s.var_dir / "agent" / job_id

    def _save(self, job: AgentJob) -> AgentJob:
        with self._lock:
            job.updated = time.time()
            d = self.job_dir(job.id)
            d.mkdir(parents=True, exist_ok=True)
            tmp_path = d / "job.json.tmp"
            final_path = d / "job.json"
            tmp_path.write_text(job.model_dump_json(indent=2))
            os.replace(tmp_path, final_path)
        return job

    def get(self, job_id: str) -> AgentJob:
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            raise KeyError(job_id)
        return AgentJob.model_validate_json(path.read_text())

    def list_jobs(self) -> list[AgentJob]:
        root = self.s.var_dir / "agent"
        jobs = [AgentJob.model_validate_json(p.read_text()) for p in root.glob("*/job.json")] if root.exists() else []
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def log(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        return read_log(self.job_dir(job_id) / "run.log", offset)

    def _note(self, job: AgentJob, text: str) -> None:
        with open(self.job_dir(job.id) / "run.log", "a") as f:
            f.write(f"\n=== [harness] {text}\n")

    def _filtered_env(self, agent_name: str) -> dict[str, str]:
        """Build environment dict with only allowed vars (filtered by env_allow patterns)."""
        agent_cfg = self.agents.get(agent_name)
        return {k: v for k, v in os.environ.items() if any(fnmatch(k, p) for p in agent_cfg.env_allow)}

    # lifecycle -----------------------------------------------------------
    def create(self, task_id: str, params: dict, extra: str = "", agent: str | None = None) -> AgentJob:
        if task_id not in self.templates:
            raise HarnessError(f"unknown task '{task_id}'")
        template = self.templates[task_id]
        clean = validate_params(template, params)
        agent_cfg = self.agents.get(agent)
        job_id = f"{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        now = time.time()
        job = AgentJob(id=job_id, task=task_id, title=template.title, params=clean, extra=extra,
                       agent=agent_cfg.name, branch=f"agent/{task_id}-{job_id}",
                       worktree=str(self.s.worktrees_dir / job_id), created=now, updated=now)
        self._save(job)
        prompt = render_prompt(template, clean, extra)
        (self.job_dir(job_id) / "prompt.md").write_text(prompt)
        return job

    def _fail(self, job: AgentJob, error: str) -> AgentJob:
        job.state, job.error = "failed", error
        self._note(job, f"FAILED: {error}")
        return self._save(job)

    def run(self, job_id: str) -> AgentJob:
        """Execute the job synchronously up to `ready_for_review` (or `failed`)."""
        job = self.get(job_id)
        template = self.templates[job.task]
        wt = Path(job.worktree)
        d = self.job_dir(job_id)
        try:
            job.state = "worktree"
            self._save(job)
            self._note(job, f"worktree {wt} on {job.branch} from {self.s.base_ref}")
            job.base_sha = gitops.create_worktree(self.s.repo, wt, job.branch, self.s.base_ref,
                                                  fetch=self.s.fetch, remote=self.s.remote)

            # Disable push and enable worktree config
            gitops.git(["config", "extensions.worktreeConfig", "true"], wt)
            gitops.git(["config", "--worktree", "remote.origin.pushurl", "DISABLED-BY-CATHLAB-HARNESS"], wt)

            job.state = "agent_running"
            self._save(job)
            prompt_file = d / "prompt.md"
            argv = build_argv(self.agents.get(job.agent), wt, prompt_file.read_text(), prompt_file)
            self._note(job, f"agent {job.agent}: {argv[0]} ...")
            env = self._filtered_env(job.agent)
            state = start_process(argv, d / "run.log", cwd=wt, timeout_s=template.timeout_min * 60,
                                  append=True, inherit_env=False, env=env).wait()
            if state != "done":
                return self._fail(job, f"agent process ended with state '{state}'")

            # Check if agent created commits (it must not)
            head_sha = gitops.git(["rev-parse", "HEAD"], wt).strip()
            if head_sha != job.base_sha:
                return self._fail(job, "agent created commits; agents must not commit")

            job.state = "guard"
            job.changed_files = gitops.stage_all(wt)
            self._save(job)
            if not job.changed_files:
                return self._fail(job, "agent made no changes")
            bad = path_violations(job.changed_files, allowed_for(template, job.params))
            if bad:
                return self._fail(job, f"changes outside allowed paths: {bad}")

            job.state = "checks"
            self._save(job)
            for check in checks_for(template, job.params):
                self._note(job, f"check '{check.name}': {check.cmd}")
                env = self._filtered_env(job.agent)
                st = start_process(check.cmd, d / "run.log", cwd=wt, timeout_s=check.timeout_s, append=True,
                                   inherit_env=False, env=env).wait()
                job.checks.append(CheckResult(name=check.name, ok=st == "done", state=st))
                self._save(job)
            failed = [c.name for c in job.checks if not c.ok]
            if failed:
                return self._fail(job, f"acceptance checks failed: {failed}")

            job.head_sha = gitops.commit(wt, f"agent({job.task}): {job.title} [{job.id}]")
            (d / "diff.patch").write_text(gitops.diff(wt, job.base_sha))
            job.state = "ready_for_review"
            self._note(job, "ready for review")
            return self._save(job)
        except Exception as e:  # noqa: BLE001 - every failure must land in the job record
            return self._fail(job, f"{type(e).__name__}: {e}")

    def start(self, job_id: str) -> threading.Thread:
        t = threading.Thread(target=self.run, args=(job_id,), daemon=True)
        with self._lock:
            self._threads[job_id] = t
        t.start()
        return t

    def diff(self, job_id: str) -> str:
        p = self.job_dir(job_id) / "diff.patch"
        return p.read_text() if p.exists() else ""

    def push(self, job_id: str) -> AgentJob:
        job = self.get(job_id)
        if job.state != "ready_for_review":
            raise HarnessError(f"job is '{job.state}', only 'ready_for_review' can be pushed")
        # Get the real URL from the main repo
        url = gitops.git(["config", "--get", "remote.origin.url"], self.s.repo).strip()
        gitops.push(Path(job.worktree), job.branch, self.s.remote, url=url)
        gitops.remove_worktree(self.s.repo, Path(job.worktree), job.branch, delete_branch=False)
        job.state = "pushed"
        self._note(job, f"pushed {job.branch}")
        return self._save(job)

    def discard(self, job_id: str, force: bool = False) -> AgentJob:
        job = self.get(job_id)
        if not force and job.state in ("agent_running", "checks", "worktree", "guard"):
            raise HarnessError(f"job is still '{job.state}'")
        if force:
            # Check if a thread is still alive for this job
            with self._lock:
                thread = self._threads.get(job_id)
            if thread is not None and thread.is_alive():
                raise HarnessError(f"job thread is still running for state '{job.state}'")
        gitops.remove_worktree(self.s.repo, Path(job.worktree), job.branch, delete_branch=True)
        job.state = "discarded"
        self._note(job, "discarded")
        return self._save(job)
