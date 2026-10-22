from pathlib import Path
import numpy as np
import pytest
from dataengine.expert import (GRIPPER_CLOSE, GRIPPER_OPEN, ExpertParams, grasp_pose, plan_waypoints,
                               run_expert_episode, waypoint_action)
from dataengine.fake_sim import FakeExpertSim
from dataengine.geometry import quat_rotate
from dataengine.variation import VariationSpec, eval_configs, object_poses, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
CFG = train_configs(SPEC)[0]


def test_side_grasp_for_bottle_approaches_along_plus_y_below_centre():
    pose = object_poses(SPEC, CFG)["bottle"]
    pos, quat, approach = grasp_pose(SPEC.objects["bottle"], pose)
    np.testing.assert_allclose(approach, [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(quat_rotate(quat, np.array([0, 0, 1.0])), approach, atol=1e-9)
    assert pos[2] == pytest.approx(pose[2] - 0.025)


def test_top_grasp_for_syringe_points_down_at_centre():
    pose = object_poses(SPEC, CFG)["syringe"]
    pos, quat, approach = grasp_pose(SPEC.objects["syringe"], pose)
    np.testing.assert_allclose(approach, [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(quat_rotate(quat, np.array([0, 0, 1.0])), [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(pos, pose[:3])


def test_waypoint_sequence_and_gripper_commands():
    wps = plan_waypoints(SPEC, "vial", object_poses(SPEC, CFG)["vial"])
    assert [w.name for w in wps] == ["pre_grasp", "grasp", "close", "lift", "above_tray", "lower", "open", "retreat"]
    assert [w.gripper for w in wps] == [GRIPPER_OPEN] * 2 + [GRIPPER_CLOSE] * 4 + [GRIPPER_OPEN] * 2
    assert np.allclose(wps[5].pos[:2], SPEC.scene.tray_center)


def test_actions_are_normalized_and_clipped():
    wps = plan_waypoints(SPEC, "vial", object_poses(SPEC, CFG)["vial"])
    a = waypoint_action(np.array([0.0, 0.0, 1.0, 1.0, 0, 0, 0]), wps[0])
    assert a.shape == (7,) and a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)


@pytest.mark.parametrize("obj", ["bottle", "vial", "syringe"])
def test_expert_succeeds_in_fake_sim_for_every_object(obj):
    cfgs = [c for c in train_configs(SPEC) if c.object == obj][:16]
    sim = FakeExpertSim(SPEC)
    results = [run_expert_episode(sim, SPEC, c) for c in cfgs]
    assert all(r.success for r in results), [r.trace for r in results if not r.success]
    assert all(r.steps < ExpertParams().max_steps for r in results)


def test_expert_succeeds_on_held_out_tier():
    sim = FakeExpertSim(SPEC)
    assert all(run_expert_episode(sim, SPEC, c).success for c in eval_configs(SPEC, "all_held_out", [5000, 5001]))


def test_record_callback_gets_every_step():
    seen = []
    r = run_expert_episode(FakeExpertSim(SPEC), SPEC, CFG, record=lambda o, a: seen.append((o["state"].shape, a.shape)))
    assert len(seen) == r.steps and seen[0] == ((8,), (7,))
