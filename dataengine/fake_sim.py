"""Kinematic fake simulator implementing ExpertSim + SimBackend for GPU-free tests and pipeline smoke runs."""
import numpy as np
from contracts import STAGES, SceneSpec, SimBackend, StepInfo
from dataengine.expert import Q_DOWN
from dataengine.geometry import quat_conj, quat_from_axis_angle, quat_mul, quat_rotate, quat_to_axis_angle
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses

HOME_POSE = np.array([0.45, 0.0, 0.40, *Q_DOWN])


class FakeExpertSim(SimBackend):
    """Each action moves the TCP exactly by action*max (metres/radians). A closed gripper picks up the target
    when the TCP is inside its bounding box (+2 cm); opening drops it straight down onto the tray or table."""

    def __init__(self, spec: VariationSpec, max_dpos: float = 0.05, max_drot: float = 0.5, image_size: int = 256):
        self.spec, self.max_dpos, self.max_drot, self.image_size = spec, max_dpos, max_drot, image_size
        self.cfg: EpisodeConfig | None = None

    # ExpertSim -----------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig, *_, **__) -> dict:
        if isinstance(cfg, SceneSpec):
            raise TypeError("FakeExpertSim.reset expects an EpisodeConfig")
        self.cfg = cfg
        self.ee = HOME_POSE.copy()
        self.closed = False
        self.poses = object_poses(self.spec, cfg)
        self.attached: str | None = None
        self.offset: np.ndarray | None = None
        self.released_in_tray: set[str] = set()
        self.reached = False
        self.lifted = False
        return self._obs()

    def get_object_pose(self, name: str) -> np.ndarray:
        return self.poses[name].copy()

    def get_ee_pose(self) -> np.ndarray:
        return self.ee.copy()

    def tray_contains(self, name: str) -> bool:
        return name in self.released_in_tray

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        a = np.asarray(action, dtype=float)
        self.ee[:3] += a[:3] * self.max_dpos
        self.ee[3:] = quat_mul(quat_from_axis_angle(a[3:6] * self.max_drot), self.ee[3:])
        target = self.cfg.object
        if np.linalg.norm(self.ee[:3] - self.poses[target][:3]) < 0.05 + self.spec.objects[target].length / 2:
            self.reached = True
        want_closed = a[6] > 0
        if want_closed and not self.closed:
            self._try_grasp()
        if not want_closed and self.closed and self.attached:
            self._release()
        self.closed = want_closed
        if self.attached:
            self.poses[self.attached][:3] = self.ee[:3] + self.offset
            bottom = self.poses[self.attached][2] - self.spec.objects[self.attached].center_z()
            self.lifted = self.lifted or bottom > 0.05
        return self._obs(), self._info()

    # SimBackend (render for policies) --------------------------------------------
    def render(self) -> np.ndarray:
        return np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

    # internals ---------------------------------------------------------------------
    def _try_grasp(self) -> None:
        for name, pose in self.poses.items():
            local = quat_rotate(quat_conj(pose[3:]), self.ee[:3] - pose[:3])
            if np.all(np.abs(local) <= self.spec.objects[name].half_extents() + 0.02):
                self.attached, self.offset = name, pose[:3] - self.ee[:3]
                return

    def _in_tray_xy(self, xy: np.ndarray) -> bool:
        c, s = np.array(self.spec.scene.tray_center), np.array(self.spec.scene.tray_inner)
        return bool(np.all(np.abs(xy - c) <= s / 2 - 0.01))

    def _release(self) -> None:
        name, obj = self.attached, self.spec.objects[self.attached]
        pose = self.poses[name]
        in_tray = self._in_tray_xy(pose[:2])
        pose[2] = (self.spec.scene.tray_floor_z if in_tray else 0.0) + obj.center_z()
        if in_tray:
            self.released_in_tray.add(name)
        self.attached = None

    def _info(self) -> StepInfo:
        t = self.cfg.object
        flags = {
            "reach": bool(self.reached),
            "grasp": self.attached == t or t in self.released_in_tray,
            "lift": bool(self.lifted),
            "transport": self._in_tray_xy(self.poses[t][:2]),
            "place": t in self.released_in_tray,
        }
        return StepInfo(stage_flags=flags, success=all(flags[s] for s in STAGES))

    def _obs(self) -> dict:
        fingers = np.array([0.0, 0.0]) if self.closed else np.array([0.04, 0.04])
        state = np.concatenate([self.ee[:3], quat_to_axis_angle(self.ee[3:]), fingers]).astype(np.float32)
        img = self.render()
        return {"state": state, "agentview": img, "eye_in_hand": img}
