"""Genesis scene for the cathlab pick task: table, static tray, bottle/vial/syringe from a VariationSpec, Menagerie
Panda with a delta-EE controller (IK + joint PD) that reproduces the robosuite OSC_POSE action convention."""
import genesis as gs
import numpy as np
from dataengine.geometry import quat_mul, quat_rotate
from dataengine.variation import ObjectSpec, VariationSpec, object_poses, train_configs

TABLE_TOP_Z = 0.8             # world z of the table top; core frame = world - CORE_ORIGIN
CORE_ORIGIN = np.array([0.0, 0.0, TABLE_TOP_Z])
BASE_HEIGHT = 0.112           # Panda base above the table top (robosuite)
TABLE_THICKNESS = 0.05
TABLE_CENTER_X = 0.56
TRAY_WALL_HEIGHT = 0.04
TRAY_WALL_THICKNESS = 0.01
TRAY_RGBA = (0.55, 0.62, 0.68, 1.0)
FLOOR_RGB = (0.45, 0.36, 0.28)
PLUNGER_RADIUS, PLUNGER_LENGTH = 0.004, 0.040
FLANGE_SIZE = (0.003, 0.030, 0.010)
HOME_QPOS = (0.0, -0.2917, 0.0, -2.5521, 0.0, 2.3877, 0.7854)
FINGER_OPEN = 0.04
TCP_OFFSET = 0.097            # robosuite grip site: 0.097 m along hand +z
# Grip-site frame = hand frame rotated -90 deg about its z (as robosuite's gripper eef site): fingers close along
# local x, approach along local z, which is what dataengine.expert.Q_DOWN assumes.
HAND_TO_EE = np.array([np.cos(np.pi / 4), 0.0, 0.0, -np.sin(np.pi / 4)])
PANDA_MJCF = "xml/franka_emika_panda/panda_no_tendon.xml"
# 5 ms (10 steps per 20 Hz action): at 10 ms the 11-20 g vial chattered at ~6 rad/s between the 20 N fingers and was
# shaken out during transport (docs/decisions.md).
SIM_DT = 0.005
ARM_DOFS = np.arange(7)
FINGER_DOFS = np.array([7, 8])
# Controller
MAX_DPOS, MAX_DROT = 0.05, 0.5           # action scaling (robosuite OSC_POSE output_max)
# robosuite OSC_POSE does not reach the commanded delta within one 20 Hz step: at steady state it moves ~25 % of the
# translation and ~13 % of the rotation per step (measured, docs/decisions.md). The same fractions are applied here so
# an action produces the same TCP motion in both engines.
POS_GAIN, ROT_GAIN = 0.235, 0.125
MAX_DQ = 0.2                             # rad per control step
IK_DAMPING = 0.02
NULLSPACE_GAIN = 0.05
# Fingers as robosuite's PandaGripper: position servo kp 1000, force limit 20 N per finger, closed target 0.
FINGER_KP, FINGER_KV, FINGER_FORCE = 1000.0, 20.0, 20.0
# Torsional friction (m, ~ contact patch radius) on the objects: Genesis contacts are points, and a side-grasped bottle
# otherwise pivots about the finger axis during transport. The patch is bounded by the 17 mm fingertip pads and by half
# the object radius (a larger value made the thin vial squirt out of the grasp).
MAX_TORSIONAL_FRICTION = 0.02
CAMERAS = ("agentview", "eye_in_hand")
AGENTVIEW = dict(pos=(1.06, 0.0, 0.55), dir=(-0.7081, 0.0, -0.7061), up=(-0.7061, 0.0, 0.7081), fov=45.0)
EYE_IN_HAND = dict(pos=(0.05, 0.0, 0.0), fov=75.0)   # hand frame; looks along hand +z, up hand +x

_GS_READY = False


def init_genesis() -> None:
    """gs.init exactly once per process."""
    global _GS_READY
    if not _GS_READY:
        gs.init(backend=gs.gpu, logging_level="warning")
        _GS_READY = True


def _to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x, "detach") else np.asarray(x, dtype=np.float64)


def syringe_mjcf(spec: ObjectSpec) -> str:
    """Lying syringe along local x (as sim/robosuite SyringeObject): barrel, flange, plunger in one free body whose
    origin is the bounding-box centre."""
    r, total = spec.radius, spec.length
    barrel = total - PLUNGER_LENGTH
    x0 = -total / 2
    fx, fy, fz = FLANGE_SIZE
    vol = np.pi * r**2 * barrel + np.pi * PLUNGER_RADIUS**2 * PLUNGER_LENGTH + fx * fy * fz
    rho = spec.mass / vol
    return f"""<mujoco model="syringe"><worldbody><body name="syringe"><freejoint/>
  <geom type="cylinder" size="{r} {barrel / 2}" pos="{x0 + barrel / 2} 0 0" quat="0.70710678 0 0.70710678 0" density="{rho}"/>
  <geom type="box" size="{fx / 2} {fy / 2} {fz / 2}" pos="{x0 + barrel + fx / 2} 0 0" density="{rho}"/>
  <geom type="cylinder" size="{PLUNGER_RADIUS} {PLUNGER_LENGTH / 2}" pos="{x0 + barrel + PLUNGER_LENGTH / 2} 0 0"
        quat="0.70710678 0 0.70710678 0" density="{rho}"/>
</body></worldbody></mujoco>"""


def look_at(pos, direction, up) -> dict:
    pos = np.asarray(pos, dtype=float)
    return dict(pos=pos, lookat=pos + np.asarray(direction, dtype=float), up=np.asarray(up, dtype=float))


class CathlabGenesisEnv:
    """Single Panda at a table with a fixed tray and the three task objects, one Genesis world (n_envs=0).

    Core frame = world - CORE_ORIGIN (robot base frame, table top at z = 0, identity base rotation). Object poses and
    the TCP are reported in the core frame, quaternions wxyz."""

    def __init__(self, spec: VariationSpec, camera_size: int = 256, render: bool = True, dt: float = SIM_DT,
                 substeps: int = 1, control_hz: float = 20.0):
        init_genesis()
        self.spec = spec
        self.render_enabled = render
        self.camera_size = camera_size
        self.dt, self.substeps = dt, substeps
        self.n_sub = max(1, round(1.0 / (control_hz * dt)))
        self.control_dt = 1.0 / control_hz
        self._arm_plan = None
        self.layout_poses = object_poses(spec, train_configs(spec)[0])
        self.nominal_light = dict(color=np.array([1.0, 1.0, 1.0]), intensity=3.0, ambient=np.array([0.35, 0.35, 0.35]))
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            rigid_options=gs.options.RigidOptions(enable_self_collision=False, enable_torsional_friction=True),
            vis_options=gs.options.VisOptions(
                ambient_light=tuple(self.nominal_light["ambient"]), shadow=True, background_color=(0.6, 0.6, 0.6),
                lights=[gs.options.vis.DirectionalLight(dir=(-0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0),
                                                        intensity=self.nominal_light["intensity"])]),
            show_viewer=False)
        self.scene = scene
        scene.add_entity(gs.morphs.Plane(), surface=gs.surfaces.Default(color=FLOOR_RGB))
        self.table = self._add_table(gs)
        self._add_tray(gs)
        self.objects = {name: self._add_object(gs, name, o) for name, o in spec.objects.items()}
        self.robot = scene.add_entity(
            gs.morphs.MJCF(file=PANDA_MJCF, pos=tuple(CORE_ORIGIN + [0, 0, BASE_HEIGHT])),
            material=gs.materials.Rigid(gravity_compensation=1.0))
        # Pedestal under the base (visual only), like robosuite's mount.
        scene.add_entity(gs.morphs.Cylinder(pos=(-0.02, 0, (TABLE_TOP_Z + BASE_HEIGHT) / 2), radius=0.1,
                                            height=TABLE_TOP_Z + BASE_HEIGHT - 0.01, fixed=True, collision=False),
                         surface=gs.surfaces.Default(color=(0.1, 0.1, 0.1)))
        self.cameras = {}
        if render:
            av = look_at(CORE_ORIGIN + AGENTVIEW["pos"], AGENTVIEW["dir"], AGENTVIEW["up"])
            self.cameras["agentview"] = scene.add_camera(res=(camera_size, camera_size), fov=AGENTVIEW["fov"],
                                                         GUI=False, **av)
            self.cameras["eye_in_hand"] = scene.add_camera(res=(camera_size, camera_size), fov=EYE_IN_HAND["fov"],
                                                           pos=(0, 0, 1), lookat=(0, 0, 0), near=0.01, GUI=False)
        scene.build()
        self.hand = self.robot.get_link("hand")
        self.finger_links = {self.robot.get_link(n).idx for n in ("left_finger", "right_finger")}
        self.home = np.array([*HOME_QPOS, FINGER_OPEN, FINGER_OPEN])
        self._tcp_local = np.array([0.0, 0.0, TCP_OFFSET])
        self.robot.set_dofs_kp(np.array([FINGER_KP, FINGER_KP]), FINGER_DOFS)
        self.robot.set_dofs_kv(np.array([FINGER_KV, FINGER_KV]), FINGER_DOFS)
        self.robot.set_dofs_force_range(np.array([-FINGER_FORCE] * 2), np.array([FINGER_FORCE] * 2), FINGER_DOFS)
        self.reset()

    # build ------------------------------------------------------------------------------
    def _add_table(self, gs):
        sx, sy = self.spec.scene.table_size
        table = self.scene.add_entity(
            gs.morphs.Box(pos=(TABLE_CENTER_X, 0, TABLE_TOP_Z - TABLE_THICKNESS / 2), size=(sx, sy, TABLE_THICKNESS),
                          fixed=True),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)))
        for dx in (-1, 1):
            for dy in (-1, 1):
                self.scene.add_entity(
                    gs.morphs.Box(pos=(TABLE_CENTER_X + dx * (sx / 2 - 0.05), dy * (sy / 2 - 0.05),
                                       (TABLE_TOP_Z - TABLE_THICKNESS) / 2),
                                  size=(0.04, 0.04, TABLE_TOP_Z - TABLE_THICKNESS), fixed=True, collision=False),
                    surface=gs.surfaces.Default(color=(0.3, 0.3, 0.3)))
        return table

    def _add_tray(self, gs):
        """Static open tray: floor of thickness tray_floor_z plus 4 walls around the inner area."""
        ix, iy = self.spec.scene.tray_inner
        t, h, fz = TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT, self.spec.scene.tray_floor_z
        cx, cy = self.spec.scene.tray_center
        ox, oy = ix / 2 + t, iy / 2 + t
        boxes = [((0, 0, fz / 2), (ix, iy, fz))]
        for s in (-1, 1):
            boxes += [((s * (ox - t / 2), 0, h / 2), (t, 2 * oy, h)), ((0, s * (oy - t / 2), h / 2), (ix, t, h))]
        surface = gs.surfaces.Default(color=TRAY_RGBA[:3])
        for pos, size in boxes:
            self.scene.add_entity(gs.morphs.Box(pos=tuple(CORE_ORIGIN + [cx + pos[0], cy + pos[1], pos[2]]),
                                                size=size, fixed=True), surface=surface)

    def _add_object(self, gs, name: str, o: ObjectSpec):
        surface = gs.surfaces.Default(color=(1.0, 1.0, 1.0))   # white vertex colours; tinted by material at reset
        pose = self.layout_poses[name]
        pos, quat = tuple(CORE_ORIGIN + pose[:3]), tuple(pose[3:])
        torsional = min(MAX_TORSIONAL_FRICTION, o.radius / 2)
        if o.shape == "syringe":
            morph = gs.morphs.MJCF(file=syringe_mjcf(o), pos=pos, quat=quat, convexify=False)
            return self.scene.add_entity(morph, material=gs.materials.Rigid(friction_torsional=torsional),
                                         surface=surface)
        rho = o.mass / (np.pi * o.radius**2 * o.length)
        return self.scene.add_entity(gs.morphs.Cylinder(pos=pos, quat=quat, radius=o.radius, height=o.length),
                                     material=gs.materials.Rigid(rho=rho, friction_torsional=torsional),
                                     surface=surface)

    # state ------------------------------------------------------------------------------
    def reset(self) -> None:
        """Robot at HOME_QPOS (PD targets too, fingers open) and objects at the layout poses, velocities zeroed."""
        self.robot.set_dofs_position(self.home, zero_velocity=True)
        self.robot.control_dofs_position(self.home)
        self._arm_plan = None
        self.set_object_poses(self.layout_poses)

    def set_object_poses(self, poses: dict[str, np.ndarray]) -> None:
        """poses: name -> [x, y, z, qw, qx, qy, qz] (object centre, core frame). Zeroes object velocities."""
        for name, pose in poses.items():
            obj = self.objects[name]
            obj.set_pos(CORE_ORIGIN + np.asarray(pose[:3]), zero_velocity=True)
            obj.set_quat(np.asarray(pose[3:]), zero_velocity=True)

    def object_pose(self, name: str) -> np.ndarray:
        obj = self.objects[name]
        return np.concatenate([_to_np(obj.get_pos()) - CORE_ORIGIN, _to_np(obj.get_quat())])

    def object_velocity(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self.objects[name]
        return _to_np(obj.get_vel()), _to_np(obj.get_ang())

    def hand_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Hand link position (core frame) and quaternion (wxyz)."""
        return _to_np(self.hand.get_pos()) - CORE_ORIGIN, _to_np(self.hand.get_quat())

    def tcp_pose(self) -> np.ndarray:
        """[x, y, z, qw, qx, qy, qz]: grip-site frame, TCP_OFFSET along hand +z (core frame)."""
        pos, quat = self.hand_pose()
        return np.concatenate([pos + quat_rotate(quat, np.array([0, 0, TCP_OFFSET])), quat_mul(quat, HAND_TO_EE)])

    def finger_qpos(self) -> np.ndarray:
        return _to_np(self.robot.get_dofs_position(FINGER_DOFS))

    def fingers_touching(self, name: str) -> bool:
        """True if both finger links are in contact with the object."""
        c = self.robot.get_contacts(with_entity=self.objects[name])
        links = set(_to_np(c["link_a"]).astype(int).tolist()) | set(_to_np(c["link_b"]).astype(int).tolist())
        return self.finger_links <= links

    # control ----------------------------------------------------------------------------
    def apply_action(self, action: np.ndarray) -> None:
        """Plan one control period from a normalized 7-D action (robosuite OSC_POSE convention): TCP delta
        [dx, dy, dz]·0.05 m and axis-angle [drx, dry, drz]·0.5 rad in the base frame about the current TCP, gripper
        -1 open / +1 close. Damped least-squares differential IK at the TCP with a weak null-space pull towards
        HOME_QPOS (OSC's posture term); `step_physics` then interpolates the joint targets over the period."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        err = np.concatenate([a[:3] * MAX_DPOS * POS_GAIN, a[3:6] * MAX_DROT * ROT_GAIN])
        q = _to_np(self.robot.get_dofs_position(ARM_DOFS))
        J = _to_np(self.robot.get_jacobian(self.hand, local_point=self._tcp_local))[:, ARM_DOFS]
        J_pinv = J.T @ np.linalg.inv(J @ J.T + IK_DAMPING**2 * np.eye(6))
        null = np.eye(7) - np.linalg.pinv(J, rcond=1e-3) @ J      # exact projector: no leak into the task
        dq = J_pinv @ err + null @ (NULLSPACE_GAIN * (self.home[:7] - q))
        self._arm_plan = (q, np.clip(dq, -MAX_DQ, MAX_DQ))
        self.set_gripper(a[6])

    def set_gripper(self, g: float) -> None:
        target = 0.0 if g > 0 else FINGER_OPEN
        self.robot.control_dofs_position(np.array([target, target]), FINGER_DOFS)

    def step_physics(self) -> None:
        """One control period: n_sub scene steps, arm targets interpolated along the planned joint delta."""
        for i in range(self.n_sub):
            if self._arm_plan is not None:
                q0, dq = self._arm_plan
                self.robot.control_dofs_position_velocity(q0 + dq * (i + 1) / self.n_sub, dq / self.control_dt,
                                                          ARM_DOFS)
            self.scene.step()
        if self._arm_plan is not None:      # hold the reached target afterwards
            q0, dq = self._arm_plan
            self.robot.control_dofs_position(q0 + dq, ARM_DOFS)
            self._arm_plan = None

    def hold(self, gripper: float = -1.0) -> None:
        """Hold the current arm PD targets for one control period."""
        self.set_gripper(gripper)
        self.step_physics()

    def settle(self, steps: int = 10) -> None:
        """Advance `steps` control periods with the arm held at HOME_QPOS and the gripper open."""
        self.robot.control_dofs_position(self.home)
        self._arm_plan = None
        for _ in range(steps):
            self.step_physics()

    # rendering --------------------------------------------------------------------------
    def render(self) -> dict[str, np.ndarray]:
        if not self.render_enabled:
            blank = np.zeros((self.camera_size, self.camera_size, 3), dtype=np.uint8)
            return {name: blank for name in CAMERAS}
        pos, quat = self.hand_pose()
        eye = CORE_ORIGIN + pos + quat_rotate(quat, np.array(EYE_IN_HAND["pos"]))
        self.cameras["eye_in_hand"].set_pose(pos=eye, lookat=eye + quat_rotate(quat, np.array([0.0, 0.0, 1.0])),
                                             up=quat_rotate(quat, np.array([1.0, 0.0, 0.0])))
        return {name: np.ascontiguousarray(cam.render(rgb=True)[0]) for name, cam in self.cameras.items()}

    def close(self) -> None:
        self.scene.destroy()
