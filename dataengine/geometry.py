"""Quaternion helpers. Convention: q = [w, x, y, z]; poses = [x, y, z, w, qx, qy, qz]."""
import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_axis_angle(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.asarray(v) / angle
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.zeros(3)
    return 2 * np.arctan2(s, q[0]) * q[1:] / s


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot_error(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """Axis-angle (base frame) rotating q_current onto q_target."""
    return quat_to_axis_angle(quat_mul(q_target, quat_conj(q_current)))
