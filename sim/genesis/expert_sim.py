"""ExpertSim adapter for Genesis: core-frame poses (base frame, table top z = 0, wxyz) and stage flags."""
import numpy as np
from contracts import STAGES, StepInfo
from dataengine.expert import Q_DOWN, grasp_pose
from dataengine.geometry import quat_to_axis_angle
from dataengine.variation import EpisodeConfig, VariationSpec
from sim.genesis.env import CathlabGenesisEnv
from sim.genesis.randomize import GenesisAxes, apply_episode_config, apply_genesis_axes, genesis_axes

REACH_DIST = 0.05
LIFT_HEIGHT = 0.05
REST_LIN_VEL = 0.02
REST_ANG_VEL = 0.5
PI_CUSP_W = 1e-6


def state_axis_angle(quat: np.ndarray) -> np.ndarray:
    """quat_to_axis_angle with a deterministic tie-break at a rotation of exactly pi (|w| < PI_CUSP_W), where q and -q
    give opposite vectors: the axis's dominant component is made positive. The Menagerie hand at HOME_QPOS sits exactly
    there, and robosuite reports (+3.135, 0, +0.2) at home."""
    q = np.asarray(quat, dtype=float)
    if abs(q[0]) < PI_CUSP_W:
        v = q[1:]
        q = np.array([0.0, *(v if v[np.argmax(np.abs(v))] >= 0 else -v)])
    return quat_to_axis_angle(q)


class GenesisExpertSim:
    """Implements dataengine.expert.ExpertSim on CathlabGenesisEnv, with the same conventions and latched stage flags
    as sim/robosuite/expert_sim.py.

    EE pose = the grip-site frame (TCP, 0.097 m along hand +z; fingers close along local x, approach local z).
    State = TCP position, hand-link axis-angle, fingers [+q1, -q2] (robosuite signs)."""

    def __init__(self, spec: VariationSpec, env: CathlabGenesisEnv | None = None, settle_steps: int = 10,
                 q_down: np.ndarray = Q_DOWN, genesis_axes: bool = False, **env_kwargs):
        self.spec = spec
        self.env = env if env is not None else CathlabGenesisEnv(spec, **env_kwargs)
        self.settle_steps = settle_steps
        self.q_down = q_down
        self.genesis_axes = genesis_axes
        self.axes: GenesisAxes | None = None
        self.cfg: EpisodeConfig | None = None
        self._tray_floor_z = spec.scene.tray_floor_z

    # ExpertSim -------------------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig) -> dict:
        self.cfg = cfg
        self.env.reset()
        apply_episode_config(self.env, self.spec, cfg)
        self.axes = genesis_axes(cfg) if self.genesis_axes else None
        apply_genesis_axes(self.env, self.axes)
        self.env.settle(self.settle_steps)
        target = cfg.object
        self._grasp_point = grasp_pose(self.spec.objects[target], self.get_object_pose(target), self.q_down)[0]
        self._latched = {"reach": False, "grasp": False, "lift": False}
        return self._obs()

    def get_object_pose(self, name: str) -> np.ndarray:
        return self.env.object_pose(name)

    def get_ee_pose(self) -> np.ndarray:
        return self.env.tcp_pose()

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        self.env.apply_action(action)
        self.env.step_physics()
        return self._obs(), self._info()

    def tray_contains(self, name: str) -> bool:
        return self._in_tray_xy(name) and self._bottom_z(name) < self._tray_floor_z + 0.02 and not self._grasped(name)

    # stage logic -------------------------------------------------------------------------
    def _grasped(self, name: str) -> bool:
        return self.env.fingers_touching(name)

    def _bottom_z(self, name: str) -> float:
        return float(self.get_object_pose(name)[2] - self.spec.objects[name].center_z())

    def _in_tray_xy(self, name: str) -> bool:
        c, s = np.array(self.spec.scene.tray_center), np.array(self.spec.scene.tray_inner)
        return bool(np.all(np.abs(self.get_object_pose(name)[:2] - c) <= s / 2))

    def _at_rest(self, name: str) -> bool:
        lin, ang = self.env.object_velocity(name)
        return bool(np.linalg.norm(lin) < REST_LIN_VEL and np.linalg.norm(ang) < REST_ANG_VEL)

    def stage_flags(self) -> dict[str, bool]:
        t = self.cfg.object
        latched = self._latched
        latched["reach"] |= bool(np.linalg.norm(self.get_ee_pose()[:3] - self._grasp_point) < REACH_DIST)
        grasped = self._grasped(t)
        latched["grasp"] |= grasped
        latched["lift"] |= latched["grasp"] and self._bottom_z(t) > LIFT_HEIGHT
        in_tray = self._in_tray_xy(t)
        return {**latched, "transport": in_tray,
                "place": in_tray and not grasped and self._at_rest(t) and self.tray_contains(t)}

    def _info(self) -> StepInfo:
        flags = self.stage_flags()
        return StepInfo(stage_flags=flags, success=all(flags[s] for s in STAGES))

    def _obs(self) -> dict:
        pos, quat = self.env.hand_pose()
        tcp = self.env.tcp_pose()
        q1, q2 = self.env.finger_qpos()
        state = np.concatenate([tcp[:3], state_axis_angle(quat), [q1, -q2]]).astype(np.float32)
        return {"state": state, **self.env.render()}
