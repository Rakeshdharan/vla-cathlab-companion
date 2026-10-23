"""Standalone agent worker: the harness API on its own port. Runs on the host (needs the agent CLI login + git)."""
from fastapi import FastAPI
from harness.api import create_agent_router
from harness.runner import Harness
from orchestrator.hosts import protect


def create_agent_app(harness: Harness, bind_host: str | None = None) -> FastAPI:
    app = protect(FastAPI(title="cathlab agent worker"), bind_host)
    app.include_router(create_agent_router(harness), prefix="/api/agent")

    @app.get("/health")
    def health() -> dict:
        return {"name": "agent", "capabilities": ["harness"], "tasks": sorted(harness.templates)}

    return app
