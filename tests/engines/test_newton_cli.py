"""Newton engine plugin (FR3 Duo): manifest and CLI work in the core env (no newton/mujoco/warp)."""
import json
import subprocess
import sys

import pytest
from engines.registry import ROOT, load_engines

STAGES = ["preview", "check-expert", "generate", "convert", "eval", "rollout"]


def test_manifest_is_fr3():
    m = load_engines()["newton"]
    assert m.io == "fr3_duo_16d" and m.embodiments == ["fr3_duo"] and m.io_profile.action_dim == 16
    assert m.backend == "sim.newton.backend:NewtonFR3Backend" and (ROOT / m.variation).exists()


def test_help_works_without_newton():
    r = subprocess.run([sys.executable, "-m", "engines.newton.cli", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("stage", STAGES)
def test_dry_run(stage, tmp_path):
    argv = [sys.executable, "-m", "engines.newton.cli", stage, "--out", str(tmp_path), "--dry-run"]
    if stage == "convert":
        argv += ["--dataset", str(tmp_path)]
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    last = json.loads(r.stdout.strip().splitlines()[-1])
    if stage == "convert":            # ctx.run_script prints the argv list in dry-run
        assert any("convert" in a for a in last)
    else:
        assert last["stage"] == stage
