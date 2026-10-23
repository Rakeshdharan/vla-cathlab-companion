"""Engine conformance: every engine whose backend imports in this env must follow the engine contract.

    MUJOCO_GL=egl conda run -n robosuite python -m pytest -q tests/engines/test_conformance.py -k robosuite

In the core env the simulators are missing, so these tests skip.
"""
import importlib
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from contracts import STAGES, StepInfo, TaskSpec, load_yaml  # noqa: E402
from engines.registry import ROOT, load_engines  # noqa: E402

ENGINES = [m for m in load_engines().values() if m.backend]


def backend_cls(m):
    module, _, cls = m.backend.partition(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        pytest.skip(f"{m.id} backend not importable in this env: {e}")


@pytest.mark.parametrize("m", ENGINES, ids=lambda m: m.id)
def test_backend_follows_io_profile(m):
    backend = backend_cls(m)()
    target = load_yaml(ROOT / m.task, TaskSpec).objects[0]
    try:
        obs = backend.reset(None, seed=5000, target_object=target)
        for key, shape in m.io_profile.obs_shapes.items():
            assert obs[key].shape == shape, key
        obs, info = backend.step(np.zeros(m.io_profile.action_dim, dtype=np.float32))
        assert isinstance(info, StepInfo) and set(info.stage_flags) == set(STAGES) and not info.success
        assert backend.render().ndim == 3
    finally:
        backend.close()


@pytest.mark.parametrize("m", ENGINES, ids=lambda m: m.id)
def test_expert_reaches_80_percent(m, tmp_path):
    backend_cls(m)
    out = tmp_path / "check_expert"
    rc = subprocess.run([sys.executable, "-m", m.entry, "check-expert", "--out", str(out), "--n", "5"],
                        cwd=ROOT).returncode
    assert rc == 0
    for obj, v in json.loads((out / "expert_check.json").read_text())["per_object"].items():
        assert v["rate"] >= 0.8, f"{obj}: {v}"
