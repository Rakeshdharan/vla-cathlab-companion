"""ManiSkill3 (SAPIEN 3 / PhysX 5) scene for the cathlab pick task, built from a VariationSpec.

World frame = core frame: robot base frame with the table top at z = 0 (the Panda base sits at z = BASE_Z).
Importing this module registers the `panda_cathlab` agent and the `CathlabPick-v1` env with ManiSkill.
"""
import numpy as np
import sapien
import torch
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import SimConfig
from dataengine.variation import ObjectSpec, VariationSpec, object_poses, train_configs

# Same home as sim/robosuite/env.py: TCP at core (0.45, 0, 0.40), gripper down.
HOME_QPOS = (0.0, -0.2917, 0.0, -2.5521, 0.0, 2.3877, 0.7854)
FINGER_OPEN = 0.04
BASE_Z = 0.112
TABLE_CENTER_X = 0.56
TABLE_THICKNESS = 0.05
FLOOR_Z = -0.8
TRAY_WALL_HEIGHT = 0.04
TRAY_WALL_THICKNESS = 0.01
TRAY_RGBA = (0.55, 0.62, 0.68, 1.0)
PLUNGER_RADIUS, PLUNGER_LENGTH = 0.004, 0.040
FLANGE_SIZE = (0.003, 0.030, 0.010)
# SAPIEN cylinders lie along local x; this turns them upright (wxyz).
Y_ROT_90 = (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0)
AGENTVIEW_EYE, AGENTVIEW_TARGET, AGENTVIEW_FOVY_DEG = (1.06, 0.0, 0.55), (0.352, 0.0, -0.156), 45.0
EYE_IN_HAND_POS, EYE_IN_HAND_FOVY_DEG = (0.05, 0.0, 0.0), 75.0
NOMINAL_AMBIENT = np.array([0.3, 0.3, 0.3])
NOMINAL_DIRECTIONAL = (((1.0, 1.0, -1.0), (1.0, 1.0, 1.0)), ((0.0, 0.0, -1.0), (1.0, 1.0, 1.0)))
CAMERAS = ("agentview", "eye_in_hand")


def _matrix_to_wxyz(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def eye_in_hand_local_pose() -> sapien.Pose:
    """SAPIEN cameras look along local +x with +z up. Ours looks along hand +z with hand +x up."""
    forward, up = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    left = np.cross(up, forward)
    return sapien.Pose(p=EYE_IN_HAND_POS, q=_matrix_to_wxyz(np.stack([forward, left, up], axis=1)))


@register_agent()
class PandaCathlab(Panda):
    """ManiSkill Panda with our action scale: pd_ee_delta_pose takes ±0.05 m and ±0.5 rad per unit action."""
    uid = "panda_cathlab"

    @property
    def _controller_configs(self):
        cfgs = super()._controller_configs
        arm = cfgs["pd_ee_delta_pose"]["arm"]
        arm.pos_lower, arm.pos_upper = -0.05, 0.05
        arm.rot_lower, arm.rot_upper = -0.5, 0.5
        return cfgs


def _material(friction: float) -> sapien.physx.PhysxMaterial:
    return sapien.physx.PhysxMaterial(static_friction=friction, dynamic_friction=friction, restitution=0.0)


def build_task_object(scene, name: str, spec: ObjectSpec, rgba, pose: np.ndarray):
    b = scene.create_actor_builder()
    mat, render = _material(1.0), sapien.render.RenderMaterial(base_color=list(rgba))
    if spec.shape == "syringe":
        r, total = spec.radius, spec.length
        barrel_len = total - PLUNGER_LENGTH
        x0 = -total / 2
        fx, fy, fz = FLANGE_SIZE
        vol = np.pi * r**2 * barrel_len + np.pi * PLUNGER_RADIUS**2 * PLUNGER_LENGTH + fx * fy * fz
        density = spec.mass / vol
        parts = [("cyl", sapien.Pose([x0 + barrel_len / 2, 0, 0]), (r, barrel_len / 2)),
                 ("box", sapien.Pose([x0 + barrel_len + fx / 2, 0, 0]), (fx / 2, fy / 2, fz / 2)),
                 ("cyl", sapien.Pose([x0 + barrel_len + PLUNGER_LENGTH / 2, 0, 0]), (PLUNGER_RADIUS, PLUNGER_LENGTH / 2))]
        for kind, p, size in parts:
            if kind == "cyl":
                b.add_cylinder_collision(p, radius=size[0], half_length=size[1], material=mat, density=density)
                b.add_cylinder_visual(p, radius=size[0], half_length=size[1], material=render)
            else:
                b.add_box_collision(p, half_size=size, material=mat, density=density)
                b.add_box_visual(p, half_size=size, material=render)
    else:
        density = spec.mass / (np.pi * spec.radius**2 * spec.length)
        p = sapien.Pose(q=Y_ROT_90)
        b.add_cylinder_collision(p, radius=spec.radius, half_length=spec.length / 2, material=mat, density=density)
        b.add_cylinder_visual(p, radius=spec.radius, half_length=spec.length / 2, material=render)
    b.initial_pose = sapien.Pose(p=pose[:3], q=pose[3:])
    return b.build(name=name)


def build_tray(scene, spec: VariationSpec):
    """Static open tray: floor of thickness tray_floor_z plus 4 walls around the inner area, resting on the table."""
    ix, iy = spec.scene.tray_inner
    t, h, fz = TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT, spec.scene.tray_floor_z
    ox, oy = ix / 2 + t, iy / 2 + t
    boxes = [((0, 0, fz / 2), (ix / 2, iy / 2, fz / 2))]
    for sign in (-1, 1):
        boxes += [((sign * (ox - t / 2), 0, h / 2), (t / 2, oy, h / 2)),
                  ((0, sign * (oy - t / 2), h / 2), (ix / 2, t / 2, h / 2))]
    b = scene.create_actor_builder()
    render = sapien.render.RenderMaterial(base_color=list(TRAY_RGBA))
    for pos, half in boxes:
        b.add_box_collision(sapien.Pose(pos), half_size=half)
        b.add_box_visual(sapien.Pose(pos), half_size=half, material=render)
    b.initial_pose = sapien.Pose(p=[*spec.scene.tray_center, 0.0])
    return b.build_static(name="tray")


def build_table(scene, spec: VariationSpec, rgba):
    sx, sy = spec.scene.table_size
    half = (sx / 2, sy / 2, TABLE_THICKNESS / 2)
    b = scene.create_actor_builder()
    b.add_box_collision(half_size=half, material=_material(1.0))
    b.add_box_visual(half_size=half, material=sapien.render.RenderMaterial(base_color=list(rgba)))
    b.initial_pose = sapien.Pose(p=[TABLE_CENTER_X, 0.0, -TABLE_THICKNESS / 2])
    return b.build_static(name="table")


@register_env("CathlabPick-v1", max_episode_steps=None)
class CathlabPickEnv(BaseEnv):
    """Panda at a table with a static tray and the three task objects. Reset options:
    `object_poses` (name -> core-frame [x, y, z, qw, qx, qy, qz]); default = layout of train config 0."""
    SUPPORTED_ROBOTS = ["panda_cathlab"]

    def __init__(self, *args, spec: VariationSpec, camera_size: int = 256, robot_uids="panda_cathlab", **kwargs):
        self.spec = spec
        self.camera_size = camera_size
        self.layout_poses = object_poses(spec, train_configs(spec)[0])
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(sim_freq=100, control_freq=20)

    @property
    def _default_sensor_configs(self):
        s, hand = self.camera_size, self.agent.robot.links_map["panda_hand"]
        agentview = sapien_utils.look_at(eye=AGENTVIEW_EYE, target=AGENTVIEW_TARGET)
        return [CameraConfig("agentview", agentview, s, s, np.deg2rad(AGENTVIEW_FOVY_DEG), 0.01, 10),
                CameraConfig("eye_in_hand", eye_in_hand_local_pose(), s, s, np.deg2rad(EYE_IN_HAND_FOVY_DEG), 0.01,
                             10, mount=hand)]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig("render_camera", sapien_utils.look_at(eye=AGENTVIEW_EYE, target=AGENTVIEW_TARGET),
                            512, 512, np.deg2rad(AGENTVIEW_FOVY_DEG), 0.01, 10)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0.0, 0.0, BASE_Z]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=FLOOR_Z)
        self.table = build_table(self.scene, self.spec, self.spec.table_colors[0])
        self.tray = build_tray(self.scene, self.spec)
        palette = self.spec.obj_colors[0]
        self.objects = {name: build_task_object(self.scene, name, o, palette[name], self.layout_poses[name])
                        for name, o in self.spec.objects.items()}

    def _load_lighting(self, options: dict):
        self.scene.set_ambient_light(NOMINAL_AMBIENT.tolist())
        for direction, color in NOMINAL_DIRECTIONAL:
            self.scene.add_directional_light(list(direction), list(color))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        qpos = torch.tensor([[*HOME_QPOS, FINGER_OPEN, FINGER_OPEN]], dtype=torch.float32).repeat(len(env_idx), 1)
        self.agent.reset(qpos)
        self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, BASE_Z]))
        self.set_object_poses(options.get("object_poses") or self.layout_poses)

    def set_object_poses(self, poses: dict[str, np.ndarray]) -> None:
        """Core-frame object centre poses (wxyz). Zeroes object velocities."""
        for name, pose in poses.items():
            actor = self.objects[name]
            pose = np.asarray(pose, dtype=np.float32)
            actor.set_pose(Pose.create_from_pq(torch.from_numpy(pose[:3])[None], torch.from_numpy(pose[3:])[None]))
            actor.set_linear_velocity(torch.zeros(1, 3))
            actor.set_angular_velocity(torch.zeros(1, 3))

    def evaluate(self) -> dict:
        return {}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
