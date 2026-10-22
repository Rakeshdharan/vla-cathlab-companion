"""robosuite scene for the cathlab pick task: table, static tray and bottle/vial/syringe built from a VariationSpec."""
import mujoco
import numpy as np
import robosuite.macros as macros
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import CompositeObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string
from dataengine.variation import ObjectSpec, VariationSpec, object_poses, train_configs

TABLE_TOP_Z = 0.8
TABLE_THICKNESS = 0.05
TRAY_WALL_HEIGHT = 0.04
TRAY_WALL_THICKNESS = 0.01
TRAY_RGBA = (0.55, 0.62, 0.68, 1.0)
# Syringe parts not covered by ObjectSpec (design spec §3): plunger Ø8 × 40 mm, flange 30 × 10 × 3 mm.
PLUNGER_RADIUS, PLUNGER_LENGTH = 0.004, 0.040
FLANGE_SIZE = (0.003, 0.030, 0.010)
CAMERAS = ("agentview", "robot0_eye_in_hand")
Y_ROT_90 = (np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0)  # wxyz, turns a cylinder's z axis onto x
# robosuite's default Panda home puts the TCP at z = 0.21 (core frame), inside the 0.25 m bottle when a layout
# places it below the hand. This home (IK, same posture family and orientation) puts the TCP at (0.45, 0, 0.40).
HOME_QPOS = (0.0, -0.2917, 0.0, -2.5521, 0.0, 2.3877, 0.7854)


class SyringeObject(CompositeObject):
    """Lying syringe along local x: barrel (ObjectSpec radius, length minus plunger), flange, plunger."""

    def __init__(self, name: str, spec: ObjectSpec, rgba):
        r, total = spec.radius, spec.length
        barrel_len = total - PLUNGER_LENGTH
        x0 = -total / 2
        fx, fy, fz = FLANGE_SIZE
        vol = np.pi * r**2 * barrel_len + np.pi * PLUNGER_RADIUS**2 * PLUNGER_LENGTH + fx * fy * fz
        super().__init__(
            name=name,
            total_size=(total / 2, fy / 2, r),
            geom_types=["cylinder", "box", "cylinder"],
            geom_sizes=[(r, barrel_len / 2), (fx / 2, fy / 2, fz / 2), (PLUNGER_RADIUS, PLUNGER_LENGTH / 2)],
            geom_locations=[(x0 + barrel_len / 2, 0, 0), (x0 + barrel_len + fx / 2, 0, 0),
                            (x0 + barrel_len + PLUNGER_LENGTH / 2, 0, 0)],
            geom_quats=[Y_ROT_90, (1, 0, 0, 0), Y_ROT_90],
            geom_names=["barrel", "flange", "plunger"],
            geom_frictions=[(1.0, 0.01, 0.001)] * 3,
            rgba=rgba,
            density=spec.mass / vol,
            solref=(0.01, 0.5),
            locations_relative_to_center=True,
            joints=[{"type": "free", "damping": "0.0001"}],
        )


def make_task_object(name: str, spec: ObjectSpec, rgba):
    if spec.shape == "syringe":
        return SyringeObject(name, spec, rgba)
    vol = np.pi * spec.radius**2 * spec.length
    return CylinderObject(name=name, size=(spec.radius, spec.length / 2), density=spec.mass / vol, rgba=rgba)


def make_tray(spec: VariationSpec) -> CompositeObject:
    """Static open tray: floor of thickness tray_floor_z plus 4 walls around the inner area."""
    ix, iy = spec.scene.tray_inner
    t, h, fz = TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT, spec.scene.tray_floor_z
    ox, oy = ix / 2 + t, iy / 2 + t
    sizes, locs = [(ix / 2, iy / 2, fz / 2)], [(0, 0, -h / 2 + fz / 2)]
    for sign in (-1, 1):
        sizes += [(t / 2, oy, h / 2), (ix / 2, t / 2, h / 2)]
        locs += [(sign * (ox - t / 2), 0, 0), (0, sign * (oy - t / 2), 0)]
    return CompositeObject(name="tray", total_size=(ox, oy, h / 2), geom_types=["box"] * 5, geom_sizes=sizes,
                           geom_locations=locs, geom_names=["floor", "wall_xn", "wall_yn", "wall_xp", "wall_yp"],
                           rgba=TRAY_RGBA, density=500.0, locations_relative_to_center=True, joints=None)


class CathlabPickPlace(ManipulationEnv):
    """Single Panda at a table with a fixed tray and the three task objects.

    Core frame = robot base frame with the table top at z = 0, i.e. world shifted by (base_x, base_y, TABLE_TOP_Z);
    the Panda base rotation is identity. `_reset_internal` places objects at `self.layout_poses` (core frame, wxyz)."""

    def __init__(self, spec: VariationSpec, robots="Panda", controller_configs=None, camera_size: int = 256,
                 **kwargs):
        self.spec = spec
        self.table_full_size = (*spec.scene.table_size, TABLE_THICKNESS)
        self.table_offset = np.array([0.0, 0.0, TABLE_TOP_Z])
        self.layout_poses = object_poses(spec, train_configs(spec)[0])
        macros.IMAGE_CONVENTION = "opencv"
        if controller_configs is None:
            controller_configs = load_composite_controller_config(controller="BASIC", robot="Panda")
        defaults = dict(control_freq=20, horizon=300, ignore_done=True, hard_reset=False, use_camera_obs=True,
                        has_renderer=False, has_offscreen_renderer=True, camera_names=list(CAMERAS),
                        camera_heights=camera_size, camera_widths=camera_size, initialization_noise=None)
        super().__init__(robots=robots, controller_configs=controller_configs, **{**defaults, **kwargs})

    def _load_robots(self):
        super()._load_robots()
        self.robots[0].init_qpos = np.array(HOME_QPOS)

    def _load_model(self):
        super()._load_model()
        robot = self.robots[0].robot_model
        xpos = robot.base_xpos_offset["table"](self.table_full_size[0])
        robot.set_base_xpos(xpos)
        self.base_xy = np.array(xpos[:2], dtype=float)

        arena = TableArena(table_full_size=self.table_full_size, table_offset=self.table_offset)
        arena.set_origin([0, 0, 0])

        palette = self.spec.obj_colors[0]
        self.objects = {name: make_task_object(name, o, palette[name]) for name, o in self.spec.objects.items()}
        self.tray = make_tray(self.spec)
        tray_pos = (*(np.asarray(self.spec.scene.tray_center) + self.base_xy), TABLE_TOP_Z + TRAY_WALL_HEIGHT / 2)
        self.tray.get_obj().set("pos", array_to_string(tray_pos))

        self.model = ManipulationTask(mujoco_arena=arena, mujoco_robots=[robot],
                                      mujoco_objects=[*self.objects.values(), self.tray])

    def _setup_references(self):
        super()._setup_references()
        m = self.sim.model
        # Lying cylinders get a single, see-sawing contact otherwise (syringe sank 2 mm and rocked).
        m._model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
        self.obj_body_ids = {n: m.body_name2id(o.root_body) for n, o in self.objects.items()}
        # Captured once: randomize.apply_episode_config scales from these, and hard_reset=False keeps edits.
        if not hasattr(self, "nominal_inertial"):
            self.nominal_inertial = {n: (float(m.body_mass[i]), m.body_inertia[i].copy())
                                     for n, i in self.obj_body_ids.items()}
            head = m._model.vis.headlight
            self.nominal_lights = {"diffuse": m.light_diffuse.copy(), "specular": m.light_specular.copy(),
                                   "head_ambient": np.array(head.ambient), "head_diffuse": np.array(head.diffuse)}

    def _reset_internal(self):
        super()._reset_internal()
        self.set_object_poses(self.layout_poses)

    @property
    def core_origin(self) -> np.ndarray:
        return np.array([*self.base_xy, TABLE_TOP_Z])

    def set_object_poses(self, poses: dict[str, np.ndarray]) -> None:
        """poses: name -> [x, y, z, qw, qx, qy, qz] (object centre, core frame). Zeroes object velocities."""
        for name, pose in poses.items():
            joint = self.objects[name].joints[0]
            self.sim.data.set_joint_qpos(joint, np.concatenate([np.asarray(pose[:3]) + self.core_origin, pose[3:]]))
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def settle(self, steps: int = 10, gripper: float = -1.0) -> None:
        """Advance `steps` control periods holding the arm still, without rendering or advancing the episode clock."""
        action = np.zeros(self.action_dim)
        action[-1] = gripper
        for _ in range(steps):
            policy_step = True
            for _ in range(int(self.control_timestep / self.model_timestep)):
                self.sim.forward()
                self._pre_action(action, policy_step)
                self.sim.step()
                policy_step = False

    def reward(self, action=None) -> float:
        return 0.0
