"""HTTP API of the agent harness (mounted at /api/agent by the control plane or the agent worker)."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from harness.model import ParamError
from harness.runner import Harness, HarnessError


class JobCreate(BaseModel):
    task: str
    params: dict = Field(default_factory=dict)
    extra: str = ""
    agent: str | None = None
    start: bool = True


def create_agent_router(harness: Harness) -> APIRouter:
    r = APIRouter()

    def job_or_404(job_id: str):
        try:
            return harness.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown agent job {job_id}") from None

    @r.get("/tasks")
    def tasks() -> list[dict]:
        return [t.model_dump() for t in harness.templates.values()]

    @r.get("/agents")
    def agents() -> dict:
        return {"default": harness.agents.default,
                "agents": [{"name": a.name, "description": a.description} for a in harness.agents.agents.values()]}

    @r.post("/jobs")
    def create(req: JobCreate) -> dict:
        try:
            job = harness.create(req.task, req.params, req.extra, req.agent)
        except (HarnessError, ParamError, KeyError) as e:
            raise HTTPException(400, str(e)) from e
        if req.start:
            harness.start(job.id)
        return job.model_dump()

    @r.get("/jobs")
    def list_jobs() -> list[dict]:
        return [j.model_dump() for j in harness.list_jobs()]

    @r.get("/jobs/{job_id}")
    def get(job_id: str) -> dict:
        return job_or_404(job_id).model_dump()

    @r.get("/jobs/{job_id}/log")
    def log(job_id: str, offset: int = 0) -> dict:
        job = job_or_404(job_id)
        text, new_offset = harness.log(job_id, offset)
        return {"text": text, "offset": new_offset, "state": job.state}

    @r.get("/jobs/{job_id}/prompt", response_class=PlainTextResponse)
    def prompt(job_id: str) -> str:
        job_or_404(job_id)
        return (harness.job_dir(job_id) / "prompt.md").read_text()

    @r.get("/jobs/{job_id}/diff", response_class=PlainTextResponse)
    def diff(job_id: str) -> str:
        job_or_404(job_id)
        return harness.diff(job_id)

    @r.post("/jobs/{job_id}/push")
    def push(job_id: str) -> dict:
        job_or_404(job_id)
        try:
            return harness.push(job_id).model_dump()
        except HarnessError as e:
            raise HTTPException(409, str(e)) from e

    @r.post("/jobs/{job_id}/discard")
    def discard(job_id: str, force: bool = False) -> dict:
        job_or_404(job_id)
        try:
            return harness.discard(job_id, force=force).model_dump()
        except HarnessError as e:
            raise HTTPException(409, str(e)) from e

    return r
