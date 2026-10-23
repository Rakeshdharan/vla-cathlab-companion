"""ExpertSim adapter for ManiSkill3: core-frame poses (= ManiSkill world frame, wxyz) and stage flags."""
import numpy as np
from contracts import STAGES, StepInfo
from dataengine.expert import Q_DOWN, grasp_pose
from dataengine.geometry import quat_mul, quat_rotate, quat_to_axis_angle
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses
from sim.maniskill.env import CAMERAS, CathlabPickEnv
from sim.maniskill.randomize import apply_episode_config

REACH_DIST = 0.05
LIFT_HEIGHT = 0.05
REST_LIN_VEL = 0.02
REST_ANG_VEL = 0.5
# robosuite's grip site (our TCP): 0.097 m along panda_hand +z, rotated -90° about hand z so the fingers close along
# TCP +x (measured in robosuite at HOME_QPOS). ManiSkill's own panda_hand_tcp is 0.1034 m out and not rotated.
TCP_OFFSET = 0.097
Q_HAND_TO_TCP = np.array([np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)])


def _np(t) -> np.ndarray:
    return t[0].detach().cpu().numpy().astype(np.float64)


class ManiSkillExpertSim:
    """Implements dataengine.expert.ExpertSim on CathlabPickEnv (CPU PhysX, one env).

    Action in: our 7-D [dpos/0.05, drot/0.5 (base frame), gripper -1 open/+1 close]; the agent's controller is scaled
    to match; the rotation and gripper signs are flipped at the boundary. State: TCP position, hand axis-angle, [+q_finger1, -q_finger2]."""

    def __init__(self, spec: VariationSpec, env: CathlabPickEnv | None = None, settle_steps: int = 10,
                 q_down: np.ndarray = Q_DOWN, render: bool = True, **env_kwargs):
        self.spec = spec
        self.render = render
        defaults = dict(obs_mode="rgb" if render else "none", control_mode="pd_ee_delta_pose", sim_backend="cpu",
                        num_envs=1)
        self.env = env if env is not None else CathlabPickEnv(spec=spec, **{**defaults, **env_kwargs})
        self.settle_steps = settle_steps
        self.q_down = q_down
        self.cfg: EpisodeConfig | None = None
        self._hand = self.env.agent.robot.links_map["panda_hand"]
        self._tray_floor_z = spec.scene.tray_floor_z
        self._blank = np.zeros((self.env.camera_size, self.env.camera_size, 3), dtype=np.uint8)

    # ExpertSim -------------------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig) -> dict:
        self.cfg = cfg
        raw, _ = self.env.reset(seed=cfg.seed, options={"object_poses": object_poses(self.spec, cfg)})
        apply_episode_config(self.env, self.spec, cfg)
        for _ in range(self.settle_steps):
            raw = self._env_step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
        if self.render and self.settle_steps == 0:
            raw = self.env.get_obs()
        target = cfg.object
        self._grasp_point = grasp_pose(self.spec.objects[target], self.get_object_pose(target), self.q_down)[0]
        self._latched = {"reach": False, "grasp": False, "lift": False}
        return self._obs(raw)

    def get_object_pose(self, name: str) -> np.ndarray:
        pose = self.env.objects[name].pose
        return np.concatenate([_np(pose.p), _np(pose.q)])

    def get_ee_pose(self) -> np.ndarray:
        p, q = _np(self._hand.pose.p), _np(self._hand.pose.q)
        return np.concatenate([p + quat_rotate(q, np.array([0.0, 0.0, TCP_OFFSET])), quat_mul(q, Q_HAND_TO_TCP)])

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        raw = self._env_step(action)
        return self._obs(raw), self._info()

    def tray_contains(self, name: str) -> bool:
        return self._in_tray_xy(name) and self._bottom_z(name) < self._tray_floor_z + 0.02 and not self._grasped(name)

    # stage logic -------------------------------------------------------------------------
    def _env_step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).copy()
        # ManiSkill 3.0.1 PDEEPoseController scales rotation by rot_lower (< 0), which flips its sign.
        a[3:6] = -a[3:6]
        a[6] = -a[6]  # ManiSkill's mimic gripper: +1 open, -1 closed
        return self.env.step(a)[0]

    def _grasped(self, name: str) -> bool:
        return bool(self.env.agent.is_grasping(self.env.objects[name])[0])

    def _bottom_z(self, name: str) -> float:
        return float(self.get_object_pose(name)[2] - self.spec.objects[name].center_z())

    def _in_tray_xy(self, name: str) -> bool:
        c, s = np.array(self.spec.scene.tray_center), np.array(self.spec.scene.tray_inner)
        return bool(np.all(np.abs(self.get_object_pose(name)[:2] - c) <= s / 2))

    def _at_rest(self, name: str) -> bool:
        actor = self.env.objects[name]
        return bool(np.linalg.norm(_np(actor.linear_velocity)) < REST_LIN_VEL
                    and np.linalg.norm(_np(actor.angular_velocity)) < REST_ANG_VEL)

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

    def state(self) -> np.ndarray:
        q_hand = _np(self._hand.pose.q)
        fingers = _np(self.env.agent.robot.get_qpos())[7:9]
        return np.concatenate([self.get_ee_pose()[:3], quat_to_axis_angle(q_hand),
                               [fingers[0], -fingers[1]]]).astype(np.float32)

    def _obs(self, raw: dict) -> dict:
        images = {c: raw["sensor_data"][c]["rgb"][0].cpu().numpy() if self.render else self._blank for c in CAMERAS}
        return {"state": self.state(), **images}
