"""contracts.SimBackend for the Newton FR3 Duo cathlab scene (io fr3_duo_16d). Import it only in ~/venvs/newton.

One Newton world (MuJoCo-Warp solver, GPU). `seed` + `tier` pick the episode with the vendored
`variation.eval_configs`, so the same seed is the same episode as in Aravind's MuJoCo/Newton eval.
"""
from pathlib import Path

import numpy as np
from contracts import STAGES, SceneSpec, SimBackend, StepInfo
# Module-level on purpose: in envs without the simulator stack the import fails and conformance skips.
from sim.newton.fr3.newton.task_scene import load_variation
import warp

CAMERAS = ("external_camera", "left_wrist_camera", "right_wrist_camera")
DEFAULT_SPEC = Path(__file__).parent / "configs/variation/cathlab_fr3_newton.yaml"


class NewtonFR3Backend(SimBackend):
    on_reset = None      # callable(cfg) after every reset, e.g. FR3ExpertPolicy.begin

    def __init__(self, spec_path: str | Path | None = None, tier: str = "in_distribution", render: bool = True):
        self.variation = load_variation(spec_path or DEFAULT_SPEC)
        self.tier, self.render_images = tier, render
        self.sim = None
        self.cfg = None

    def config_for(self, seed: int, target_object: str):
        from sim.newton.fr3.newton.variation import eval_configs
        cfgs = [c for c in eval_configs(self.variation, self.tier, [seed]) if c.object == target_object]
        if not cfgs:
            raise ValueError(f"unknown object '{target_object}'; expected one of {list(self.variation.objects)}")
        return cfgs[0]

    def reset(self, scene: SceneSpec | None, seed: int, target_object: str) -> dict[str, np.ndarray]:
        return self.reset_config(self.config_for(seed, target_object))

    def reset_config(self, cfg) -> dict[str, np.ndarray]:
        from sim.newton.fr3.newton.task_stages import StageTracker
        if self.sim is None:
            from sim.newton.fr3.newton.scene import NewtonTaskSim   # lazy: needs newton + a CUDA GPU
            self.sim = NewtonTaskSim(1, variation=self.variation, render=self.render_images)
        self.cfg = cfg
        self.sim.reset([cfg])
        self.tracker = StageTracker(1)
        self._nan = False
        if self.on_reset is not None:
            self.on_reset(cfg)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]:
        a = np.asarray(action, np.float32).reshape(1, 16)
        state = self.sim.step(a)
        self._nan |= bool(np.isnan(state).any())
        success = bool(self.tracker.update(self.sim.flags())[0]) and not self._nan
        flags = {s: bool(self.tracker.flags[s][0]) for s in STAGES}
        return self._obs(), StepInfo(stage_flags=flags, success=success)

    def _obs(self) -> dict[str, np.ndarray]:
        obs = {"state": self.sim.state()[0]}
        if self.render_images:
            imgs = self.sim.images()
            obs.update({c: imgs[c][0] for c in CAMERAS})
        return obs

    def render(self) -> np.ndarray:
        return self.sim.images()["external_camera"][0]

    def close(self) -> None:
        if self.sim is not None:
            warp.synchronize()
        self.sim = None
