from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel


class PolicyMeta(BaseModel):
    name: str
    params_m: float
    chunk_size: int
    control_hz: float
    action_dim: int


class PolicyAdapter(ABC):
    @abstractmethod
    def load(self, ckpt: str) -> None: ...

    @abstractmethod
    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        """Return an action chunk of shape (chunk_size, action_dim)."""

    @abstractmethod
    def meta(self) -> PolicyMeta: ...


class RandomPolicy(PolicyAdapter):
    def __init__(self, action_dim: int = 16, chunk_size: int = 10, seed: int = 0):
        self._meta = PolicyMeta(name="random", params_m=0.0, chunk_size=chunk_size,
                                control_hz=30.0, action_dim=action_dim)
        self._rng = np.random.default_rng(seed)

    def load(self, ckpt: str) -> None:
        pass

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        return self._rng.uniform(-1, 1, (self._meta.chunk_size, self._meta.action_dim))

    def meta(self) -> PolicyMeta:
        return self._meta
