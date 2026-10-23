"""ManiSkill adapter conventions (run in the maniskill env; skipped elsewhere)."""
import numpy as np
import pytest

pytest.importorskip("mani_skill")

from dataengine.geometry import quat_yaw  # noqa: E402
from dataengine.variation import VariationSpec, object_poses, train_configs  # noqa: E402
from engines.registry import ROOT  # noqa: E402
from sim.maniskill.expert_sim import ManiSkillExpertSim  # noqa: E402

SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


@pytest.fixture(scope="module")
def sim():
    s = ManiSkillExpertSim(SPEC, render=False)
    yield s
    s.env.close()


def test_home_matches_robosuite(sim):
    obs = sim.reset(train_configs(SPEC)[0])
    # robosuite at HOME_QPOS: state [0.45, 0, 0.40, 3.135, 0, 0.200, 0.0387, -0.0387], grip-site quat below.
    np.testing.assert_allclose(obs["state"][:6], [0.45, 0.0, 0.40, 3.135, 0.0, 0.200], atol=3e-3)
    assert obs["state"][6] > 0.03 and obs["state"][7] < -0.03
    np.testing.assert_allclose(np.abs(sim.get_ee_pose()[3:]), [0.045, 0.7055, 0.7058, 0.045], atol=3e-3)


def test_objects_rest_after_settle(sim):
    cfg = train_configs(SPEC)[0]
    sim.reset(cfg)
    for name, pose in object_poses(SPEC, cfg).items():
        assert np.linalg.norm(sim.get_object_pose(name)[:3] - pose[:3]) < 0.002, name


def test_action_signs(sim):
    sim.reset(train_configs(SPEC)[0])
    e0 = sim.get_ee_pose()
    for _ in range(3):
        sim.step(np.array([1, 0, 0, 0, 0, 1, -1], dtype=np.float32))
    e1 = sim.get_ee_pose()
    assert e1[0] - e0[0] > 0.03
    dyaw = np.angle(np.exp(1j * (quat_yaw(e1[3:]) - quat_yaw(e0[3:]))))
    assert dyaw > 0.3
    for _ in range(10):
        obs, _ = sim.step(np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    assert obs["state"][6] < 0.01 and obs["state"][7] > -0.01
