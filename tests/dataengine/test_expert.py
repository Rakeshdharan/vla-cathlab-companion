from pathlib import Path
import numpy as np
import pytest
from dataengine.expert import (GRIPPER_CLOSE, GRIPPER_OPEN, Q_DOWN, ExpertParams, grasp_pose, plan_waypoints,
                               run_expert_episode, waypoint_action)
from dataengine.fake_sim import FakeExpertSim
from dataengine.geometry import quat_conj, quat_mul, quat_rotate, quat_yaw, yaw_quat
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
    assert [w.name for w in wps] == ["approach", "pre_grasp", "grasp", "close", "lift", "above_tray", "lower", "open",
                                     "retreat"]
    assert [w.gripper for w in wps] == [GRIPPER_OPEN] * 3 + [GRIPPER_CLOSE] * 4 + [GRIPPER_OPEN] * 2
    assert np.allclose(wps[6].pos[:2], SPEC.scene.tray_center)


@pytest.mark.parametrize("obj", ["bottle", "vial", "syringe"])
def test_horizontal_moves_happen_at_transit_height(obj):
    params = ExpertParams()
    wps = {w.name: w for w in plan_waypoints(SPEC, obj, object_poses(SPEC, CFG)[obj], params)}
    np.testing.assert_allclose(wps["approach"].pos[:2], wps["pre_grasp"].pos[:2])
    for name in ("approach", "lift", "above_tray"):
        assert wps[name].pos[2] >= params.transit_height - 1e-9
    np.testing.assert_allclose(wps["lift"].pos[:2], wps["grasp"].pos[:2])


def test_min_grasp_tcp_z_raises_low_grasps_only():
    poses = object_poses(SPEC, CFG)
    params = ExpertParams(min_grasp_tcp_z=0.013)
    syringe = {w.name: w for w in plan_waypoints(SPEC, "syringe", poses["syringe"], params)}
    assert syringe["grasp"].pos[2] == pytest.approx(0.013)
    bottle = {w.name: w for w in plan_waypoints(SPEC, "bottle", poses["bottle"], params)}
    assert bottle["grasp"].pos[2] == pytest.approx(poses["bottle"][2] - 0.025)


def test_rise_first_when_starting_below_transit_height():
    params = ExpertParams()
    ee = np.array([0.45, 0.0, 0.23, *quat_mul(yaw_quat(np.pi / 2), Q_DOWN)])
    wps = plan_waypoints(SPEC, "bottle", object_poses(SPEC, CFG)["bottle"], params, ee_pose=ee)
    assert [w.name for w in wps[:2]] == ["rise", "approach"]
    np.testing.assert_allclose(wps[0].pos, [0.45, 0.0, params.transit_height])
    np.testing.assert_allclose(wps[0].quat, ee[3:])
    high = np.array([0.45, 0.0, 0.40, *ee[3:]])
    assert plan_waypoints(SPEC, "bottle", object_poses(SPEC, CFG)["bottle"], params, ee_pose=high)[0].name == "approach"


def _yaw_from_down(quat):
    return quat_yaw(quat_mul(quat, quat_conj(Q_DOWN)))


def test_top_grasp_yaw_follows_reference_for_round_vial():
    pose = object_poses(SPEC, CFG)["vial"]
    ref = quat_mul(yaw_quat(0.7), Q_DOWN)
    _, quat, _ = grasp_pose(SPEC.objects["vial"], pose, ref_quat=ref)
    assert _yaw_from_down(quat) == pytest.approx(0.7)


@pytest.mark.parametrize("obj_yaw", [-3.0, -1.2, 0.4, 2.9])
def test_syringe_grasp_picks_symmetric_yaw_nearest_reference(obj_yaw):
    pose = np.array([0.5, 0.1, 0.0075, *yaw_quat(obj_yaw)])
    ref = quat_mul(yaw_quat(np.pi / 2), Q_DOWN)
    _, quat, _ = grasp_pose(SPEC.objects["syringe"], pose, ref_quat=ref)
    yaw = _yaw_from_down(quat)
    assert abs(np.angle(np.exp(1j * (yaw - np.pi / 2)))) <= np.pi / 2 + 1e-9
    # fingers still close perpendicular to the barrel
    assert abs(np.cos(yaw - (obj_yaw + np.pi / 2))) == pytest.approx(1.0)


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
