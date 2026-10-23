"""The vendored FR3 scripted expert as a PolicyAdapter (16-D joint targets, 30 Hz).

The expert is kinematic and open-loop: `begin(cfg)` plans the whole episode on a CPU MuJoCo twin of the
same scene (sim.newton.fr3.newton.cpu_sim), `predict` replays it one frame at a time in the engine.
"""
import numpy as np
from contracts import PolicyAdapter, PolicyMeta


class FR3ExpertPolicy(PolicyAdapter):
    def __init__(self, variation=None):
        self.variation = variation
        self._cpu = None
        self._plan = np.zeros((0, 16), np.float32)
        self._t = 0
        self._meta = PolicyMeta(name="expert", params_m=0.0, chunk_size=1, control_hz=30.0, action_dim=16)

    def load(self, ckpt: str = "") -> None:
        pass

    def plan(self, cfg) -> np.ndarray:
        from sim.newton.fr3.newton.cpu_sim import CpuTaskSim   # lazy: needs mujoco
        from sim.newton.fr3.newton.expert_fr3 import TaskExpert
        if self._cpu is None:
            self._cpu = CpuTaskSim(self.variation)
        self._cpu.reset(cfg)
        return np.asarray(TaskExpert(self._cpu.model, self._cpu.variation).plan(self._cpu.data, self._cpu.task_obs(),
                                                                                 cfg.object), np.float32)

    def begin(self, cfg) -> None:
        self._plan, self._t = self.plan(cfg), 0

    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        if not len(self._plan):
            raise RuntimeError("FR3ExpertPolicy.begin(cfg) must be called after every reset")
        a = self._plan[min(self._t, len(self._plan) - 1)]   # hold the last (retreat) target
        self._t += 1
        return a[None, :]

    def meta(self) -> PolicyMeta:
        return self._meta
