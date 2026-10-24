"""Drive the expert over EpisodeConfigs, record successful episodes into a sink, write metadata + manifest."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import numpy as np
from contracts import DatasetManifest, TaskSpec, save_yaml
from dataengine.expert import ExpertParams, ExpertSim, Q_DOWN, run_expert_episode
from dataengine.quality import base_provenance, is_noop
from dataengine.variation import EpisodeConfig, VariationSpec


class EpisodeSink(Protocol):
    def add_frame(self, frame: dict) -> None: ...
    def save_episode(self) -> None: ...
    def discard_episode(self) -> None: ...
    def finalize(self) -> None: ...


class MemorySink:
    """In-memory sink for tests and dry runs."""

    def __init__(self):
        self.episodes: list[list[dict]] = []
        self._buf: list[dict] = []
        self.finalized = False

    def add_frame(self, frame: dict) -> None:
        self._buf.append(frame)

    def save_episode(self) -> None:
        self.episodes.append(self._buf)
        self._buf = []

    def discard_episode(self) -> None:
        self._buf = []

    def finalize(self) -> None:
        self.finalized = True


def build_frame(obs: dict, action: np.ndarray, instruction: str) -> dict:
    return {
        "observation.state": np.asarray(obs["state"], dtype=np.float32),
        "observation.images.agentview": obs["agentview"],
        "observation.images.eye_in_hand": obs["eye_in_hand"],
        "action": np.asarray(action, dtype=np.float32),
        "task": instruction,
    }


def instruction_for(task: TaskSpec, cfg: EpisodeConfig) -> str:
    return task.instructions(cfg.object)[cfg.seed % len(task.instruction_templates)]


@dataclass
class GenerationReport:
    attempted: int = 0
    succeeded: int = 0
    per_object: dict[str, list[int]] = field(default_factory=dict)  # object -> [attempted, succeeded]

    @property
    def success_rate(self) -> float:
        return self.succeeded / max(self.attempted, 1)


def generate_scripted(sim: ExpertSim, spec: VariationSpec, task: TaskSpec, configs: list[EpisodeConfig],
                      sink: EpisodeSink, out_dir: str | Path, source_sim: str, dataset_id: str, version: str,
                      created_by: str, params: ExpertParams = ExpertParams(),
                      q_down: np.ndarray = Q_DOWN, drop_noops: bool = True,
                      extra_meta: dict | None = None) -> tuple[GenerationReport, DatasetManifest]:
    """Keep only successful episodes (success held `params.success_hold` steps), drop no-op frames, and write
    <out_dir>/meta/episodes_extra.jsonl (+ frames, dropped_noops, provenance = git commit + `extra_meta`) and
    <out_dir>/manifest.yaml."""
    out_dir = Path(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    report = GenerationReport()
    provenance = {**base_provenance(), **(extra_meta or {})}
    with open(out_dir / "meta" / "episodes_extra.jsonl", "a") as meta:
        for cfg in configs:
            instruction = instruction_for(task, cfg)
            kept = {"frames": 0, "dropped": 0, "prev": None}

            def record(obs, action, instruction=instruction, kept=kept):
                noop = drop_noops and is_noop(action, kept["prev"])
                kept["prev"] = np.asarray(action)
                if noop:
                    kept["dropped"] += 1
                    return
                kept["frames"] += 1
                sink.add_frame(build_frame(obs, action, instruction))

            result = run_expert_episode(sim, spec, cfg, params, q_down, record=record)
            counts = report.per_object.setdefault(cfg.object, [0, 0])
            counts[0] += 1
            report.attempted += 1
            if result.success:
                sink.save_episode()
                counts[1] += 1
                meta.write(json.dumps({"episode_index": report.succeeded, "source_sim": source_sim,
                                       "instruction": instruction, "steps": result.steps,
                                       "frames": kept["frames"], "dropped_noops": kept["dropped"],
                                       "config": cfg.model_dump(), "provenance": provenance}) + "\n")
                report.succeeded += 1
            else:
                sink.discard_episode()
    sink.finalize()
    manifest = DatasetManifest(dataset_id=dataset_id, version=version, source="scripted", task_id=task.task_id,
                               scene_id=f"{spec.variation_id}_{source_sim}", n_episodes=report.succeeded,
                               created_by=created_by,
                               notes=f"expert success {report.succeeded}/{report.attempted}; per object {report.per_object}")
    save_yaml(manifest, out_dir / "manifest.yaml")
    return report, manifest
