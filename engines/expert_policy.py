"""The scripted waypoint expert as a PolicyAdapter: the reference row of every eval, in any engine.

It reads poses from the engine's ExpertSim, so the engine calls `begin(cfg)` after every reset.
"""
import numpy as np
from contracts import PolicyAdapter, PolicyMeta
from dataengine.expert import ExpertParams, ExpertSim, _reached, plan_waypoints, waypoint_action
from dataengine.variation import EpisodeConfig, VariationSpec


class ExpertPolicy(PolicyAdapter):
    def __init__(self, sim: ExpertSim, spec: VariationSpec, params: ExpertParams | None = None,
                 control_hz: float = 20.0):
        self.sim, self.spec = sim, spec
        self.params = params or ExpertParams()
        self._meta = PolicyMeta(name="expert", params_m=0.0, chunk_size=1, control_hz=control_hz, action_dim=7)
        self._waypoints: list = []
        self._idx = self._held = 0

    def load(self, ckpt: str = "") -> None:
        pass

    def begin(self, cfg: EpisodeConfig) -> None:
        self._waypoints = plan_waypoints(self.spec, cfg.object, self.sim.get_object_pose(cfg.object), self.params,
                                         ee_pose=self.sim.get_ee_pose())
        self._idx = self._held = 0

    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        if not self._waypoints:
            raise RuntimeError("ExpertPolicy.begin(cfg) must be called after every reset")
        ee = self.sim.get_ee_pose()
        wp = self._waypoints[self._idx]
        # Same advance rule as dataengine.expert.run_expert_episode; the last waypoint (retreat) is held.
        if self._idx < len(self._waypoints) - 1 and _reached(ee, wp, self.params):
            if self._held >= wp.hold:
                self._idx, self._held = self._idx + 1, 0
                wp = self._waypoints[self._idx]
            else:
                self._held += 1
        return waypoint_action(ee, wp, self.params)[None, :]

    def meta(self) -> PolicyMeta:
        return self._meta
