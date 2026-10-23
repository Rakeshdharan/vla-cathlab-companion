"""SimBackend for policy evaluation in robosuite, on the same EpisodeConfigs the expert uses."""
from pathlib import Path
import numpy as np
from contracts import SceneSpec, SimBackend, StepInfo
from dataengine.variation import TIERS, EpisodeConfig, VariationSpec, eval_configs
from sim.robosuite.expert_sim import RobosuiteExpertSim

DEFAULT_SPEC = Path(__file__).resolve().parents[2] / "configs/variation/cathlab_v1.yaml"


class RobosuiteBackend(SimBackend):
    """`reset(scene, seed, target_object)` runs `eval_configs(spec, tier, [seed])` for the target, so a seed means the
    same episode in every simulator. `scene` is not used; set the tier with the constructor or `.tier`."""

    def __init__(self, spec: VariationSpec | None = None, tier: str = "in_distribution",
                 expert_sim: RobosuiteExpertSim | None = None, **env_kwargs):
        if tier not in TIERS:
            raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")
        self.spec = spec if spec is not None else VariationSpec.load(DEFAULT_SPEC)
        self.tier = tier
        self.sim = expert_sim if expert_sim is not None else RobosuiteExpertSim(self.spec, **env_kwargs)
        self._obs: dict | None = None

    def config_for(self, seed: int, target_object: str) -> EpisodeConfig:
        for cfg in eval_configs(self.spec, self.tier, [seed]):
            if cfg.object == target_object:
                return cfg
        raise ValueError(f"unknown target object {target_object!r}; expected one of {list(self.spec.objects)}")

    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]:
        return self.reset_config(self.config_for(seed, target_object))

    def reset_config(self, cfg: EpisodeConfig) -> dict[str, np.ndarray]:
        self._obs = self.sim.reset(cfg)
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]:
        self._obs, info = self.sim.step(action)
        return self._obs, info

    def render(self) -> np.ndarray:
        return self._obs["agentview"]

    def close(self) -> None:
        self.sim.env.close()
