"""Worker app: runs stage commands inside one Python environment (robosuite, lerobot, isaac, ...).

Start: python -m orchestrator.worker --name robosuite --capabilities generate,eval --port 8711
"""
import argparse
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from orchestrator.process import ProcessHandle, read_log, start_process


class JobRequest(BaseModel):
    argv: list[str]
    name: str = "job"
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None


def create_worker_app(name: str, capabilities: list[str], var_dir: str | Path) -> FastAPI:
    var_dir = Path(var_dir)
    app = FastAPI(title=f"cathlab worker: {name}")
    jobs: dict[str, dict] = {}

    def view(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        h: ProcessHandle = job["handle"]
        return {"id": job_id, "name": job["name"], "state": h.state, "returncode": h.returncode,
                "started": job["started"]}

    @app.get("/health")
    def health() -> dict:
        running = sum(1 for j in jobs.values() if j["handle"].state == "running")
        return {"name": name, "capabilities": capabilities, "python": platform.python_version(),
                "executable": sys.executable, "pid": os.getpid(), "jobs_running": running}

    @app.post("/jobs")
    def submit(req: JobRequest) -> dict:
        if not req.argv:
            raise HTTPException(400, "argv must not be empty")
        job_id = uuid.uuid4().hex[:12]
        handle = start_process(req.argv, var_dir / "jobs" / f"{job_id}.log", cwd=req.cwd, env=req.env,
                               timeout_s=req.timeout_s)
        jobs[job_id] = {"handle": handle, "name": req.name, "started": time.time()}
        return view(job_id)

    @app.get("/jobs/{job_id}")
    def get(job_id: str) -> dict:
        return view(job_id)

    @app.get("/jobs/{job_id}/log")
    def log(job_id: str, offset: int = 0) -> dict:
        state = view(job_id)["state"]
        text, new_offset = read_log(jobs[job_id]["handle"].log_path, offset)
        return {"text": text, "offset": new_offset, "state": state}

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        view(job_id)
        jobs[job_id]["handle"].cancel()
        return view(job_id)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--capabilities", default="", help="comma-separated, e.g. generate,eval")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--var-dir", default="var/workers")
    args = ap.parse_args()
    caps = [c for c in args.capabilities.split(",") if c]
    app = create_worker_app(args.name, caps, Path(args.var_dir) / args.name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
