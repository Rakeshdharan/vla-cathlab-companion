"""Sim-agnostic scripted expert: grasp templates -> waypoints -> normalized 7-D EEF-delta actions."""
from dataclasses import dataclass, field
from typing import Callable, Protocol
import numpy as np
from contracts import StepInfo
from dataengine.geometry import quat_from_axis_angle, quat_mul, quat_rotate, quat_yaw, rot_error, yaw_quat
from dataengine.variation import EpisodeConfig, ObjectSpec, VariationSpec

# Panda hand pointing straight down (approach axis = -z base): 180° about x.
Q_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_OPEN, GRIPPER_CLOSE = -1.0, 1.0


class ExpertSim(Protocol):
    """What each simulator adapter must provide to run the expert. Obs must contain
    'state' (8,), 'agentview' (256,256,3) uint8 and 'eye_in_hand' (256,256,3) uint8."""

    def reset(self, cfg: EpisodeConfig) -> dict: ...
    def get_object_pose(self, name: str) -> np.ndarray: ...   # [x,y,z,qw,qx,qy,qz], object centre, base frame
    def get_ee_pose(self) -> np.ndarray: ...                  # [x,y,z,qw,qx,qy,qz], fingertip centre (TCP)
    def step(self, action: np.ndarray) -> tuple[dict, StepInfo]: ...
    def tray_contains(self, name: str) -> bool: ...


@dataclass
class ExpertParams:
    max_dpos: float = 0.05     # metres per unit action (robosuite OSC_POSE default output_max)
    max_drot: float = 0.5      # radians per unit action
    pos_tol: float = 0.008
    rot_tol: float = 0.08
    close_steps: int = 10
    open_steps: int = 8
    max_steps: int = 300
    approach_offset: float = 0.10
    lift_height: float = 0.15
    tray_clearance: float = 0.20
    release_height: float = 0.02


@dataclass
class Waypoint:
    pos: np.ndarray
    quat: np.ndarray
    gripper: float
    hold: int = 0      # extra steps to stay here after reaching it (lets the gripper actuate)
    name: str = ""


@dataclass
class ExpertResult:
    success: bool
    steps: int
    final_info: StepInfo | None
    waypoints_reached: int
    trace: list[str] = field(default_factory=list)


def grasp_pose(obj: ObjectSpec, obj_pose: np.ndarray, q_down: np.ndarray = Q_DOWN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tcp_position, tcp_quat, approach_direction) for the object's grasp template."""
    centre, yaw = obj_pose[:3], quat_yaw(obj_pose[3:])
    if obj.grasp == "side":
        # Side grasp at 40% of the height, approaching along +y, fingers closing horizontally.
        pos = centre + np.array([0.0, 0.0, -0.1 * obj.length])
        quat = quat_mul(quat_from_axis_angle(np.array([np.pi / 2, 0.0, 0.0])), q_down)
        return pos, quat, quat_rotate(quat, np.array([0.0, 0.0, 1.0]))
    # Top-down grasp; fingers close perpendicular to the object's long (local x) axis.
    quat = quat_mul(yaw_quat(yaw + np.pi / 2), q_down) if not obj.upright else quat_mul(yaw_quat(yaw), q_down)
    return centre.copy(), quat, np.array([0.0, 0.0, -1.0])


def plan_waypoints(spec: VariationSpec, target: str, obj_pose: np.ndarray, params: ExpertParams = ExpertParams(),
                   q_down: np.ndarray = Q_DOWN) -> list[Waypoint]:
    obj = spec.objects[target]
    g_pos, g_quat, approach = grasp_pose(obj, obj_pose, q_down)
    bottom_z = obj_pose[2] - obj.center_z()
    grasp_above_bottom = g_pos[2] - bottom_z
    tray = np.array([*spec.scene.tray_center, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    place_z = spec.scene.tray_floor_z + params.release_height + grasp_above_bottom
    above_tray = tray + up * (place_z + params.tray_clearance)
    lowered = tray + up * place_z
    return [
        Waypoint(g_pos - approach * params.approach_offset, g_quat, GRIPPER_OPEN, name="pre_grasp"),
        Waypoint(g_pos, g_quat, GRIPPER_OPEN, name="grasp"),
        Waypoint(g_pos, g_quat, GRIPPER_CLOSE, hold=params.close_steps, name="close"),
        Waypoint(g_pos + up * params.lift_height, g_quat, GRIPPER_CLOSE, name="lift"),
        Waypoint(above_tray, g_quat, GRIPPER_CLOSE, name="above_tray"),
        Waypoint(lowered, g_quat, GRIPPER_CLOSE, name="lower"),
        Waypoint(lowered, g_quat, GRIPPER_OPEN, hold=params.open_steps, name="open"),
        Waypoint(lowered + up * params.lift_height, g_quat, GRIPPER_OPEN, name="retreat"),
    ]


def waypoint_action(ee_pose: np.ndarray, wp: Waypoint, params: ExpertParams = ExpertParams()) -> np.ndarray:
    """Normalized 7-D action in [-1, 1]: [dpos/max_dpos, drot/max_drot, gripper]."""
    dpos = np.clip((wp.pos - ee_pose[:3]) / params.max_dpos, -1.0, 1.0)
    drot = np.clip(rot_error(wp.quat, ee_pose[3:]) / params.max_drot, -1.0, 1.0)
    return np.concatenate([dpos, drot, [wp.gripper]]).astype(np.float32)


def _reached(ee_pose: np.ndarray, wp: Waypoint, params: ExpertParams) -> bool:
    return (np.linalg.norm(wp.pos - ee_pose[:3]) < params.pos_tol
            and np.linalg.norm(rot_error(wp.quat, ee_pose[3:])) < params.rot_tol)


def run_expert_episode(sim: ExpertSim, spec: VariationSpec, cfg: EpisodeConfig,
                       params: ExpertParams = ExpertParams(), q_down: np.ndarray = Q_DOWN,
                       record: Callable[[dict, np.ndarray], None] | None = None) -> ExpertResult:
    """Run one scripted episode. `record(obs, action)` is called before every step with the obs the action was chosen from."""
    obs = sim.reset(cfg)
    waypoints = plan_waypoints(spec, cfg.object, sim.get_object_pose(cfg.object), params, q_down)
    steps, idx, held, info, trace = 0, 0, 0, None, []
    while idx < len(waypoints) and steps < params.max_steps:
        wp = waypoints[idx]
        action = waypoint_action(sim.get_ee_pose(), wp, params)
        if record is not None:
            record(obs, action)
        obs, info = sim.step(action)
        steps += 1
        if _reached(sim.get_ee_pose(), wp, params):
            if held >= wp.hold:
                trace.append(wp.name)
                idx, held = idx + 1, 0
            else:
                held += 1
    success = bool(info is not None and info.success and sim.tray_contains(cfg.object))
    return ExpertResult(success=success, steps=steps, final_info=info, waypoints_reached=idx, trace=trace)
