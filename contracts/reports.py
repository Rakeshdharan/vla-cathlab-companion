import json
import math
from pathlib import Path
from pydantic import BaseModel
from contracts.specs import STAGES, Stage


class EpisodeResult(BaseModel):
    episode: int
    seed: int
    object: str
    instruction: str
    success: bool
    failed_stage: Stage | None
    steps: int
    latency_ms_p50: float
    latency_ms_p95: float
    scene_variant: str = "default"
    video_path: str | None = None
    judge_verdict: str | None = None


class EvalReport(BaseModel):
    model_version: str
    dataset_versions: list[str]
    backend: str
    episodes: list[EpisodeResult]

    def success_rate(self) -> float:
        return sum(e.success for e in self.episodes) / max(len(self.episodes), 1)

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        n = len(self.episodes)
        if n == 0:
            return (0.0, 0.0)
        p = self.success_rate()
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def stage_failure_hist(self) -> dict[str, int]:
        hist = {s: 0 for s in STAGES}
        for e in self.episodes:
            if e.failed_stage:
                hist[e.failed_stage] += 1
        return hist

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        return cls.model_validate(json.loads(Path(path).read_text()))


class ModelCard(BaseModel):
    model_name: str
    version: str
    adapter: str
    ckpt_uri: str
    dataset_versions: list[str]
    eval_report_path: str
    wandb_run: str | None = None
    git_sha: str | None = None
    promoted: bool = False
