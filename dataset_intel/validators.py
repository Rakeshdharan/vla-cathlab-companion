"""Per-episode validators (numpy only). Each returns a list of Issue for one episode, or for all episodes."""
import numpy as np
from dataset_intel.model import Issue

GRIPPER = {"panda_libero": (6, "gt", 0.0), "fr3_duo_16d": (15, "lt", 0.5)}   # action index, closed when ... value
GRIPPER_DIMS = {"panda_libero": [6], "fr3_duo_16d": [7, 15]}                # excluded from action_jump (binary)


def gripper_closed(actions: np.ndarray, io: str) -> np.ndarray:
    idx, op, thr = GRIPPER[io]
    g = actions[:, idx]
    return g > thr if op == "gt" else g < thr


def check_nan(ep: int, state: np.ndarray, actions: np.ndarray) -> list[Issue]:
    bad = [name for name, a in (("state", state), ("action", actions)) if not np.all(np.isfinite(a))]
    return [Issue(episode=ep, kind="nan", detail=f"non-finite values in {', '.join(bad)}", severity="error")] if bad else []


def check_frozen(ep: int, state: np.ndarray, fps: int, seconds: float, eps: float) -> list[Issue]:
    """Longest run of consecutive steps with max |Δstate| <= eps. Gripper holds move the fingers (part of the state),
    so only a stalled episode reaches `seconds`."""
    if len(state) < 2:
        return []
    still = np.max(np.abs(np.diff(state, axis=0)), axis=1) <= eps
    best = run = 0
    for s in still:
        run = run + 1 if s else 0
        best = max(best, run)
    if best + 1 >= seconds * fps:
        return [Issue(episode=ep, kind="frozen", detail=f"state unchanged for {(best + 1) / fps:.1f} s",
                      severity="warning")]
    return []


def check_action_jump(ep: int, actions: np.ndarray, io: str, threshold: float) -> list[Issue]:
    if len(actions) < 2:
        return []
    arm = np.delete(actions, GRIPPER_DIMS[io], axis=1)
    jumps = np.max(np.abs(np.diff(arm, axis=0)), axis=1)
    i = int(np.argmax(jumps))
    if jumps[i] > threshold:
        return [Issue(episode=ep, kind="action_jump", detail=f"step {i + 1}: max |Δaction| {jumps[i]:.2f} > {threshold}",
                      severity="warning")]
    return []


def check_gripper(ep: int, actions: np.ndarray, io: str) -> list[Issue]:
    if not np.any(gripper_closed(actions, io)):
        return [Issue(episode=ep, kind="gripper_never_closes", detail="gripper command never closes",
                      severity="error")]
    return []


def idle_fraction(actions: np.ndarray, io: str, eps: float = 1e-4) -> float:
    """No-op frames: panda_libero = LIBERO rule (no EEF motion, gripper unchanged); joint I/O = target unchanged."""
    if len(actions) == 0:
        return 0.0
    if io == "panda_libero":
        still = np.linalg.norm(actions[:, :-1], axis=1) < eps
        same_grip = np.concatenate([[False], actions[1:, -1] == actions[:-1, -1]])
        return float(np.mean(still & same_grip))
    if len(actions) < 2:
        return 0.0
    return float(np.mean(np.max(np.abs(np.diff(actions, axis=0)), axis=1) < eps))


def check_idle(ep: int, frac: float, max_frac: float) -> list[Issue]:
    if frac > max_frac:
        return [Issue(episode=ep, kind="idle_fraction", detail=f"{frac:.0%} no-op frames > {max_frac:.0%}",
                      severity="warning")]
    return []


def check_frame_counts(ep: int, length: int, video_frames: dict[str, int]) -> list[Issue]:
    bad = {k: n for k, n in video_frames.items() if n != length}
    if bad:
        detail = ", ".join(f"{k}: {n}" for k, n in bad.items())
        return [Issue(episode=ep, kind="frame_count_mismatch", detail=f"{length} parquet frames vs video {detail}",
                      severity="error")]
    return []


def length_outliers(lengths: dict[int, int], k: float) -> list[Issue]:
    if len(lengths) < 4:
        return []
    vals = np.array(list(lengths.values()))
    q1, q3 = np.percentile(vals, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return [Issue(episode=ep, kind="length_outlier", detail=f"length {n} outside [{lo:.0f}, {hi:.0f}]",
                  severity="warning") for ep, n in lengths.items() if n < lo or n > hi]
