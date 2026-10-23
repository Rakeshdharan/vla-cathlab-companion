"""Standalone agent worker: the harness API on its own port. Runs on the host (needs the agent CLI login + git)."""
from fastapi import FastAPI
from harness.api import create_agent_router
from harness.runner import Harness


def create_agent_app(harness: Harness) -> FastAPI:
    app = FastAPI(title="cathlab agent worker")
    app.include_router(create_agent_router(harness), prefix="/api/agent")

    @app.get("/health")
    def health() -> dict:
        return {"name": "agent", "capabilities": ["harness"], "tasks": sorted(harness.templates)}

    return app
