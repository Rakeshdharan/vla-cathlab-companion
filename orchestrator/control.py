"""Control plane: catalog, runs, jobs, worker health, agent harness, and the UI.

Start (dev): python -m orchestrator.control --port 8700   ->  open http://localhost:8700
"""
import argparse
import os
from pathlib import Path
import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from orchestrator.catalog import load_catalog, selection_problems
from orchestrator.store import Store

ROOT = Path(__file__).resolve().parents[1]


class SelectionBody(BaseModel):
    selection: dict
    note: str = ""


class WorkerEntry(BaseModel):
    name: str
    url: str


class WorkersConfig(BaseModel):
    workers: list[WorkerEntry] = Field(default_factory=list)


def load_workers(path: str | Path) -> WorkersConfig:
    p = Path(path)
    return WorkersConfig.model_validate(yaml.safe_load(p.read_text())) if p.exists() else WorkersConfig()


def worker_health(entry: WorkerEntry, timeout: float = 1.0) -> dict:
    try:
        headers = {}
        if token := os.environ.get("CATHLAB_WORKER_TOKEN"):
            headers["X-Cathlab-Token"] = token
        r = httpx.get(entry.url.rstrip("/") + "/health", timeout=timeout, headers=headers)
        r.raise_for_status()
        return {"name": entry.name, "url": entry.url, "healthy": True, **r.json(), "error": ""}
    except Exception as e:  # noqa: BLE001 - any failure means "offline"
        return {"name": entry.name, "url": entry.url, "healthy": False, "capabilities": [], "error": str(e)[:200]}


def create_control_app(repo_root: str | Path = ROOT, var_dir: str | Path | None = None,
                       catalog_path: str | Path | None = None, workers_path: str | Path | None = None,
                       frontend_dir: str | Path | None = None, harness=None) -> FastAPI:
    repo_root = Path(repo_root)
    var_dir = Path(var_dir) if var_dir else repo_root / "var"
    catalog_path = Path(catalog_path or repo_root / "configs/catalog.yaml")
    workers_path = Path(workers_path or os.environ.get("CATHLAB_WORKERS", repo_root / "configs/workers.yaml"))
    frontend_dir = Path(frontend_dir or repo_root / "frontend")
    store = Store(var_dir / "control.db")
    app = FastAPI(title="CathLab Control Station")
    app.state.store = store

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "repo": str(repo_root), "agent": harness is not None}

    @app.get("/api/catalog")
    def catalog() -> dict:
        return load_catalog(catalog_path).model_dump()

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
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return r.model_dump()

    @app.get("/api/jobs")
    def jobs(kind: str | None = None, run_id: str | None = None) -> list[dict]:
        return [j.model_dump() for j in store.list_jobs(kind=kind, run_id=run_id)]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = store.get_job(job_id)
        if j is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return j.model_dump()

    @app.get("/api/workers")
    def workers() -> list[dict]:
        return [worker_health(w) for w in load_workers(workers_path).workers]

    if harness is not None:
        from harness.api import create_agent_router
        app.include_router(create_agent_router(harness), prefix="/api/agent")

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
    uvicorn.run(create_control_app(ROOT, harness=harness), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
