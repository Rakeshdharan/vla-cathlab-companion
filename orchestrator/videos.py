"""Recorded rollout videos for the demo: ASSIST -> RECORDED ROLLOUTS. Two sources: a config of pre-recorded
videos (configs/demo_videos.yaml, e.g. HF URLs uploaded by the GPU session) and every finished rollout job
on this machine (its rollout.json + video, newest first)."""
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

MAX_LOCAL = 20


class VideoEntry(BaseModel):
    label: str
    engine: str = ""
    object: str = ""
    model: str = ""
    success: bool | None = None
    url: str | None = None
    path: str | None = None

    @model_validator(mode="after")
    def _one_of_url_or_path(self):
        if bool(self.url) == bool(self.path):
            raise ValueError("exactly one of url or path is required")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return self

    def view(self) -> dict:
        return {**self.model_dump(), "src": self.url or f"/api/videos/file?path={self.path}"}


class VideosConfig(BaseModel):
    videos: list[VideoEntry] = Field(default_factory=list)


def load_videos(path: str | Path) -> list[VideoEntry]:
    p = Path(path)
    return VideosConfig.model_validate(yaml.safe_load(p.read_text())).videos if p.exists() else []


def local_rollouts(store, var_dir: str | Path, limit: int = MAX_LOCAL) -> list[dict]:
    var_dir = Path(var_dir)
    jobs = [j for j in store.list_jobs(kind="stage", limit=500) if j.name == "rollout" and j.state == "done"][:limit]
    out = []
    for job in jobs:
        out_dir = job.result.get("out")
        if not out_dir:
            continue
        rollout_path = Path(out_dir) / "rollout.json"
        try:
            rollout = json.loads(rollout_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        video = rollout.get("video")
        video_url = f"/api/jobs/{job.id}/file?path={video}" if video else None
        out.append({"job_id": job.id, "run_id": job.run_id, "object": rollout.get("object"),
                    "policy": rollout.get("policy"), "instruction": rollout.get("instruction"),
                    "success": rollout.get("success"), "steps": rollout.get("steps"),
                    "failed_stage": rollout.get("failed_stage"), "video_url": video_url})
    return out
