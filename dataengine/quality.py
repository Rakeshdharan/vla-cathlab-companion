"""Data-quality rules shared by every engine's generator and converter (spec P6)."""
import platform
import subprocess
from pathlib import Path

import numpy as np

NOOP_EPS = 1e-4
ROOT = Path(__file__).resolve().parents[1]


def is_noop(action: np.ndarray, prev_action: np.ndarray | None, eps: float = NOOP_EPS) -> bool:
    """LIBERO's rule: no translation/rotation and the gripper command unchanged from the previous action."""
    still = float(np.linalg.norm(np.asarray(action[:-1], dtype=np.float64))) < eps
    return still if prev_action is None else still and float(action[-1]) == float(prev_action[-1])


def base_provenance() -> dict:
    """Git commit (+ dirty flag for the data code) and Python, recorded with every generated episode."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "sim", "engines", "dataengine"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip()) if commit else None
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"git_commit": commit, "git_dirty": dirty, "python": platform.python_version()}


def engine_provenance(engine: str, **versions) -> dict:
    """base_provenance() + this engine's name and version fields (e.g. robosuite/mujoco, mani_skill/sapien/torch,
    genesis/torch), all under the one `provenance` dict every engine's generator passes as `extra_meta` to
    dataengine.generate.generate_scripted (data-quality rule E4: uniform shape across engines)."""
    return {**base_provenance(), "engine": engine, **versions}
