"""ExpertSim adapter for robosuite: core-frame poses (base frame, table top z = 0, wxyz) and stage flags."""
import mujoco
import numpy as np
from contracts import STAGES, StepInfo
from dataengine.expert import Q_DOWN, grasp_pose
from dataengine.geometry import quat_to_axis_angle
from dataengine.variation import EpisodeConfig, VariationSpec
from sim.robosuite.env import CathlabPickPlace
from sim.robosuite.randomize import apply_episode_config

REACH_DIST = 0.05
LIFT_HEIGHT = 0.05
REST_LIN_VEL = 0.02
REST_ANG_VEL = 0.5


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]])


class RobosuiteExpertSim:
    """Implements dataengine.expert.ExpertSim on CathlabPickPlace.

    EE pose = the gripper grip site (TCP, the frame OSC controls). In that frame the fingers close along local x and
    the approach is local z, which is what the core's Q_DOWN assumes, so no q_down override is needed.
    State orientation uses the hand body quaternion (robosuite `robot0_eef_quat`), as LIBERO does."""

    def __init__(self, spec: VariationSpec, env: CathlabPickPlace | None = None, settle_steps: int = 10,
                 q_down: np.ndarray = Q_DOWN, **env_kwargs):
        self.spec = spec
        self.env = env if env is not None else CathlabPickPlace(spec, **env_kwargs)
        self.settle_steps = settle_steps
        self.q_down = q_down
        self.cfg: EpisodeConfig | None = None
        robot = self.env.robots[0]
        self._gripper = robot.gripper
        self._site_id = robot.eef_site_id["right"]
        self._tray_floor_z = spec.scene.tray_floor_z

    # ExpertSim -------------------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig) -> dict:
        self.cfg = cfg
        self.env.reset()
        apply_episode_config(self.env, self.spec, cfg)
        self.env.settle(self.settle_steps)
        target = cfg.object
        self._grasp_point = grasp_pose(self.spec.objects[target], self.get_object_pose(target), self.q_down)[0]
        self._latched = {"reach": False, "grasp": False, "lift": False}
        return self._obs(self.env._get_observations(force_update=True))

    def get_object_pose(self, name: str) -> np.ndarray:
        d, i = self.env.sim.data, self.env.obj_body_ids[name]
        return np.concatenate([d.body_xpos[i] - self.env.core_origin, d.body_xquat[i]])

    def get_ee_pose(self) -> np.ndarray:
        d = self.env.sim.data
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self._site_id])
        return np.concatenate([d.site_xpos[self._site_id] - self.env.core_origin, quat])

    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]:
        raw, _, _, _ = self.env.step(np.asarray(action, dtype=np.float64))
        return self._obs(raw), self._info()

    def tray_contains(self, name: str) -> bool:
        return self._in_tray_xy(name) and self._bottom_z(name) < self._tray_floor_z + 0.02 and not self._grasped(name)

    # stage logic -------------------------------------------------------------------------
    def _grasped(self, name: str) -> bool:
        return bool(self.env._check_grasp(gripper=self._gripper, object_geoms=self.env.objects[name]))

    def _bottom_z(self, name: str) -> float:
        return float(self.get_object_pose(name)[2] - self.spec.objects[name].center_z())

    def _in_tray_xy(self, name: str) -> bool:
        c, s = np.array(self.spec.scene.tray_center), np.array(self.spec.scene.tray_inner)
        return bool(np.all(np.abs(self.get_object_pose(name)[:2] - c) <= s / 2))

    def _at_rest(self, name: str) -> bool:
        m, d = self.env.sim.model, self.env.sim.data
        start, end = m.get_joint_qvel_addr(self.env.objects[name].joints[0])
        qvel = d.qvel[start:end]
        return bool(np.linalg.norm(qvel[:3]) < REST_LIN_VEL and np.linalg.norm(qvel[3:]) < REST_ANG_VEL)

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

    def _obs(self, raw: dict) -> dict:
        state = np.concatenate([raw["robot0_eef_pos"] - self.env.core_origin,
                                quat_to_axis_angle(xyzw_to_wxyz(raw["robot0_eef_quat"])),
                                raw["robot0_gripper_qpos"]]).astype(np.float32)
        blank = None if self.env.use_camera_obs else np.zeros((self.env.camera_heights[0], self.env.camera_widths[0], 3),
                                                              dtype=np.uint8)
        return {"state": state,
                "agentview": raw["agentview_image"] if blank is None else blank,
                "eye_in_hand": raw["robot0_eye_in_hand_image"] if blank is None else blank}
