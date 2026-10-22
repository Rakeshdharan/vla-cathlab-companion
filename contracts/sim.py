from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel
from contracts.specs import STAGES, SceneSpec


class StepInfo(BaseModel):
    stage_flags: dict[str, bool]
    success: bool
    collision: bool = False


class SimBackend(ABC):
    @abstractmethod
    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]: ...

    @abstractmethod
    def render(self) -> np.ndarray: ...

    def close(self) -> None:
        pass


class DummySim(SimBackend):
    """Deterministic fake: completes one stage every `steps_per_stage` steps, optionally stalls at `fail_at`."""

    def __init__(self, steps_per_stage: int = 5, fail_at: str | None = None):
        self.steps_per_stage = steps_per_stage
        self.fail_at = fail_at
        self._t = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {"state": np.zeros(16, dtype=np.float32), "image_head": self.render()}

    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]:
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]:
        self._t += 1
        done = min(self._t // self.steps_per_stage, len(STAGES))
        if self.fail_at is not None:
            done = min(done, STAGES.index(self.fail_at))
        flags = {s: i < done for i, s in enumerate(STAGES)}
        return self._obs(), StepInfo(stage_flags=flags, success=all(flags.values()))

    def render(self) -> np.ndarray:
        return np.zeros((224, 224, 3), dtype=np.uint8)
