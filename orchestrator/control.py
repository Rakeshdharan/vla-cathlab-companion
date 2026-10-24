"""Control plane: catalog, runs, jobs, worker health, agent harness, and the UI.

Start (dev): python -m orchestrator.control --port 8700   ->  open http://localhost:8700
"""
import argparse
import os
from pathlib import Path
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from engines.registry import ENGINES_DIR, load_engines
from orchestrator.catalog import load_catalog, selection_problems
from orchestrator.hosts import protect
from orchestrator.crossphysics import TABLE as CROSS_PHYSICS_TABLE, load_cross_physics
from orchestrator.datasets import load_datasets, local_manifests
from orchestrator.videos import load_videos, local_rollouts
from orchestrator.loop import create_loop_router
from orchestrator.loop_stages import make_local_actions
from orchestrator.assist import create_assist_router
from serve.registry import Registry
from orchestrator.pipeline import Pipeline, PipelineError, worker_request
from orchestrator.store import Store

ROOT = Path(__file__).resolve().parents[1]


class SelectionBody(BaseModel):
    selection: dict
    note: str = ""


class StageStart(BaseModel):
    params: dict = Field(default_factory=dict)


class WorkerEntry(BaseModel):
    name: str
    url: str


class WorkersConfig(BaseModel):
    workers: list[WorkerEntry] = Field(default_factory=list)


def load_workers(path: str | Path) -> WorkersConfig:
    p = Path(path)
    return WorkersConfig.model_validate(yaml.safe_load(p.read_text())) if p.exists() else WorkersConfig()


def create_control_app(repo_root: str | Path = ROOT, var_dir: str | Path | None = None,
                       catalog_path: str | Path | None = None, workers_path: str | Path | None = None,
                       frontend_dir: str | Path | None = None, harness=None, bind_host: str | None = None,
                       engines_dir: str | Path | None = None, worker_http=None,
                       eval_roots: list | None = None, datasets_path: str | Path | None = None,
                       videos_path: str | Path | None = None) -> FastAPI:
    repo_root = Path(repo_root)
    var_dir = Path(var_dir) if var_dir else repo_root / "var"
    catalog_path = Path(catalog_path or repo_root / "configs/catalog.yaml")
    datasets_path = Path(datasets_path or repo_root / "configs/datasets.yaml")
    videos_path = Path(videos_path or repo_root / "configs/demo_videos.yaml")
    workers_path = Path(workers_path or os.environ.get("CATHLAB_WORKERS", repo_root / "configs/workers.yaml"))
    frontend_dir = Path(frontend_dir or repo_root / "frontend")
    store = Store(var_dir / "control.db")
    app = protect(FastAPI(title="CathLab Control Station"), bind_host)
    app.state.store = store
    engines_dir = Path(engines_dir or ENGINES_DIR)
    registry = Registry(var_dir / "registry", repo_root=repo_root)
    app.state.registry = registry
    pipeline = Pipeline(store, var_dir, repo_root, workers_fn=lambda: load_workers(workers_path).workers,
                        engines_fn=lambda: load_engines(engines_dir), http=worker_http or worker_request,
                        resolve_policy=registry.policy_for)
    pipeline.local_actions = make_local_actions(pipeline, registry, repo_root)
    app.state.pipeline = pipeline

    def run_or_404(run_id: str):
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return r

    def job_or_404(job_id: str):
        j = store.get_job(job_id)
        if j is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return j

    def pipeline_call(fn, *args):
        try:
            return fn(*args)
        except PipelineError as e:
            raise HTTPException(e.status, str(e)) from e

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "repo": str(repo_root), "agent": harness is not None}

    @app.get("/api/catalog")
    def catalog() -> dict:
        return load_catalog(catalog_path).model_dump()

    @app.get("/api/cross-physics")
    def cross_physics() -> dict:
        return load_cross_physics(repo_root / CROSS_PHYSICS_TABLE)

    @app.get("/api/datasets")
    def datasets() -> dict:
        return {"published": [d.view() for d in load_datasets(datasets_path).datasets],
                "local": local_manifests(var_dir)}

    @app.get("/api/videos")
    def videos() -> dict:
        return {"recorded": [v.view() for v in load_videos(videos_path)], "local": local_rollouts(store, var_dir)}

    @app.get("/api/videos/file")
    def video_file(path: str):
        full = (var_dir / path).resolve()
        if var_dir.resolve() not in full.parents:
            raise HTTPException(400, "path is outside var/")
        allowed_suffixes = {'.mp4', '.webm', '.gif', '.png', '.jpg', '.jpeg'}
        if full.suffix.lower() not in allowed_suffixes:
            raise HTTPException(400, "only video/image files can be served")
        if not full.is_file():
            raise HTTPException(404, f"no file '{path}'")
        return FileResponse(full)

    @app.post("/api/selection/check")
    def check(body: SelectionBody) -> dict:
        return {"problems": [p.model_dump() for p in selection_problems(load_catalog(catalog_path), body.selection)]}

    @app.post("/api/runs")
    def create_run(body: SelectionBody) -> dict:
        problems = selection_problems(load_catalog(catalog_path), body.selection)
        if problems:
            raise HTTPException(409, {"message": "selection is not runnable", "problems": [p.model_dump() for p in problems]})
        run = store.create_run(body.selection, body.note)
        run_dir = var_dir / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run.model_dump(), sort_keys=False))
        return run.model_dump()

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return [r.model_dump() for r in store.list_runs()]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        return run_or_404(run_id).model_dump()

    @app.get("/api/jobs")
    def jobs(kind: str | None = None, run_id: str | None = None) -> list[dict]:
        return [j.model_dump() for j in store.list_jobs(kind=kind, run_id=run_id)]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return pipeline.refresh(job_or_404(job_id)).model_dump()

    @app.get("/api/workers")
    def workers() -> list[dict]:
        return pipeline.workers()

    @app.get("/api/engines")
    def engines() -> list[dict]:
        return [m.model_dump() for m in load_engines(engines_dir).values()]

    @app.get("/api/runs/{run_id}/stages")
    def run_stages(run_id: str) -> list[dict]:
        return pipeline.stages(run_or_404(run_id))

    @app.post("/api/runs/{run_id}/stages/{stage_id}")
    def start_stage(run_id: str, stage_id: str, body: StageStart) -> dict:
        return pipeline_call(pipeline.start, run_or_404(run_id), stage_id, body.params).model_dump()

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = 0) -> dict:
        return pipeline.log(job_or_404(job_id), offset)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> dict:
        return pipeline_call(pipeline.cancel, job_or_404(job_id)).model_dump()

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str) -> list[dict]:
        return pipeline_call(pipeline.files, job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/file")
    def job_file(job_id: str, path: str):
        return FileResponse(pipeline_call(pipeline.file, job_or_404(job_id), path))

    if harness is not None:
        from harness.api import create_agent_router
        app.include_router(create_agent_router(harness), prefix="/api/agent")

    roots = [Path(p) for p in eval_roots] if eval_roots else [repo_root / "reports", var_dir / "runs"]
    app.include_router(create_loop_router(store, registry, var_dir, repo_root, catalog_path, roots), prefix="/api")
    app.include_router(create_assist_router(store, pipeline, repo_root), prefix="/api/assist")

    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--no-agent", action="store_true", help="don't mount the agent harness (e.g. inside Docker)")
    args = ap.parse_args()
    harness = None
    if not args.no_agent:
        from harness.factory import default_harness
        harness = default_harness(ROOT)
    uvicorn.run(create_control_app(ROOT, harness=harness, bind_host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
