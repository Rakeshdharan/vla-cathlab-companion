import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402

pytest.importorskip("robosuite")

from engines.registry import ROOT  # noqa: E402
from engines.robosuite.evaluate import run_eval, run_rollout  # noqa: E402

SPEC = ROOT / "configs/variation/cathlab_v1.yaml"
TASK = ROOT / "configs/tasks/cathlab_pick.yaml"
SUITE = ROOT / "configs/eval/suite_v2.yaml"


def test_expert_eval_is_a_strong_reference(tmp_path):
    report = run_eval(SUITE, "expert", tmp_path, SPEC, TASK, tiers=["in_distribution"], max_seeds=1, video="failures")
    assert len(report.episodes) == 3 and report.success_rate() == 1.0
    assert {e.scene_variant for e in report.episodes} == {"in_distribution"}
    saved = json.loads((tmp_path / "eval_report.json").read_text())
    assert saved["backend"] == "robosuite" and len(saved["episodes"]) == 3


def test_random_rollout_writes_video_and_json(tmp_path):
    result = run_rollout("random", tmp_path, SPEC, TASK, "vial", max_steps=20)
    assert result["success"] is False and result["steps"] == 20
    assert (tmp_path / result["video"]).exists()
    assert json.loads((tmp_path / "rollout.json").read_text())["object"] == "vial"
