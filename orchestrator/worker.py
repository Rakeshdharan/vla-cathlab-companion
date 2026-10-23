"""Worker app: runs stage commands inside one Python environment (robosuite, lerobot, isaac, ...).

Start (localhost, no auth):
  python -m orchestrator.worker --name robosuite --capabilities generate,eval --port 8711

Start (remote host, requires auth):
  CATHLAB_WORKER_TOKEN=s3cret python -m orchestrator.worker --name robosuite --capabilities generate,eval --host 0.0.0.0 --port 8711
"""
import argparse
import os
import platform
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel, Field
from orchestrator.process import ProcessHandle, read_log, start_process


class JobRequest(BaseModel):
    argv: list[str]
    name: str = "job"
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None


def check_bind(host: str, token: str | None) -> None:
    """Raise SystemExit if binding to a non-local host without a token."""
    local_hosts = {"127.0.0.1", "localhost", "::1"}
    if host not in local_hosts and not token:
        raise SystemExit(f"refusing to bind {host} without a token: set CATHLAB_WORKER_TOKEN or --token")


def create_worker_app(name: str, capabilities: list[str], var_dir: str | Path, token: str | None = None,
                      max_jobs: int = 200) -> FastAPI:
    var_dir = Path(var_dir)
    app = FastAPI(title=f"cathlab worker: {name}")
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def _get_token_checker():
        """FastAPI dependency for token authentication."""
        def check_token(x_cathlab_token: str | None = Header(None)) -> None:
            if token:
                if x_cathlab_token is None or not secrets.compare_digest(x_cathlab_token, token):
                    raise HTTPException(401, "missing or invalid X-Cathlab-Token")
        return check_token

    token_checker = _get_token_checker()

    def _get(job_id: str) -> dict:
        """Get job record, raise 404 if not found."""
        with jobs_lock:
            job = jobs.get(job_id)
            if job is None:
                raise HTTPException(404, f"unknown job {job_id}")
            return job

    def view(job_id_or_record: str | dict) -> dict:
        """Build job view from job_id (lookup) or record (cached)."""
        if isinstance(job_id_or_record, str):
            job = _get(job_id_or_record)
            job_id = job_id_or_record
        else:
            job = job_id_or_record
            job_id = None
        h: ProcessHandle = job["handle"]
        return {"id": job_id or job.get("id"), "name": job["name"], "state": h.state, "returncode": h.returncode,
                "started": job["started"]}

    @app.get("/health")
    def health() -> dict:
        with jobs_lock:
            running = sum(1 for j in jobs.values() if j["handle"].state == "running")
        return {"name": name, "capabilities": capabilities, "python": platform.python_version(),
                "executable": sys.executable, "pid": os.getpid(), "jobs_running": running,
                "auth": token is not None}

    @app.post("/jobs")
    def submit(req: JobRequest, _: None = Depends(token_checker)) -> dict:
        if not req.argv:
            raise HTTPException(400, "argv must not be empty")
        
        with jobs_lock:
            # Evict oldest non-running jobs if at capacity
            if len(jobs) >= max_jobs:
                non_running = [(jid, j) for jid, j in jobs.items() if j["handle"].state != "running"]
                if not non_running:
                    raise HTTPException(429, "too many running jobs")
                non_running.sort(key=lambda x: x[1]["started"])
                while len(jobs) >= max_jobs and non_running:
                    jid, _ = non_running.pop(0)
                    del jobs[jid]
            
            # Reserve slot in jobs dict
            job_id = uuid.uuid4().hex[:12]
            # Placeholder to reserve the slot
            jobs[job_id] = {"handle": None, "name": req.name, "started": time.time(), "id": job_id}
        
        # Start process outside lock
        handle = start_process(req.argv, var_dir / "jobs" / f"{job_id}.log", cwd=req.cwd, env=req.env,
                               timeout_s=req.timeout_s)
        
        with jobs_lock:
            jobs[job_id]["handle"] = handle
        
        return view(job_id)

    @app.get("/jobs/{job_id}")
    def get(job_id: str, _: None = Depends(token_checker)) -> dict:
        job = _get(job_id)
        return view(job)

    @app.get("/jobs/{job_id}/log")
    def log(job_id: str, offset: int = 0, _: None = Depends(token_checker)) -> dict:
        job = _get(job_id)
        state = view(job)["state"]
        text, new_offset = read_log(job["handle"].log_path, offset)
        return {"text": text, "offset": new_offset, "state": state}

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str, _: None = Depends(token_checker)) -> dict:
        job = _get(job_id)
        job["handle"].cancel()
        return view(job)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--capabilities", default="", help="comma-separated, e.g. generate,eval")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--var-dir", default="var/workers")
    ap.add_argument("--token", default=None, help="shared auth token; defaults to CATHLAB_WORKER_TOKEN")
    args = ap.parse_args()
    token = args.token or os.environ.get("CATHLAB_WORKER_TOKEN")
    check_bind(args.host, token)
    caps = [c for c in args.capabilities.split(",") if c]
    app = create_worker_app(args.name, caps, Path(args.var_dir) / args.name, token=token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
