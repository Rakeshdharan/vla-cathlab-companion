"""Datasets for the DATA tab: published ones from configs/datasets.yaml, local ones from the DatasetManifests that
generate jobs write under var/runs/."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from contracts import DatasetManifest, load_yaml

MAX_LOCAL = 200


class DatasetEntry(BaseModel):
    repo_id: str
    revision: str | None = None
    embodiment: str
    io: str
    engine: str
    physics: str = ""
    episodes: int
    frames: int
    fps: int
    visibility: Literal["public", "private", "local"] = "public"
    local_path: str | None = None
    notes: str = ""

    def view(self) -> dict:
        url = None if self.visibility == "local" else f"https://huggingface.co/datasets/{self.repo_id}"
        return {**self.model_dump(), "url": url}


class DatasetsConfig(BaseModel):
    datasets: list[DatasetEntry] = Field(default_factory=list)


def load_datasets(path: str | Path) -> DatasetsConfig:
    p = Path(path)
    return DatasetsConfig.model_validate(yaml.safe_load(p.read_text())) if p.exists() else DatasetsConfig()


def local_manifests(var_dir: Path) -> list[dict]:
    """Manifests under var/runs/<run>/... (newest first), skipping files that are not a DatasetManifest."""
    runs = Path(var_dir) / "runs"
    paths = sorted(runs.rglob("manifest.yaml"), key=lambda p: p.stat().st_mtime, reverse=True) if runs.is_dir() else []
    out = []
    for p in paths[:MAX_LOCAL]:
        try:
            m = load_yaml(p, DatasetManifest)
        except Exception:  # noqa: BLE001 - other YAML files may share the name
            continue
        rel = p.parent.relative_to(var_dir)
        out.append({**m.model_dump(), "path": rel.as_posix(), "run_id": rel.parts[1], "modified": p.stat().st_mtime})
    return out
