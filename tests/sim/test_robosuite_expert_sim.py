import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("robosuite")

from dataengine.expert import run_expert_episode  # noqa: E402
from dataengine.geometry import quat_yaw  # noqa: E402
from dataengine.variation import VariationSpec, object_poses, train_configs  # noqa: E402
from sim.robosuite.expert_sim import RobosuiteExpertSim  # noqa: E402
from sim.robosuite.params import ROBOSUITE_EXPERT_PARAMS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
CFG = train_configs(SPEC)[0]


@pytest.fixture(scope="module")
def sim():
    s = RobosuiteExpertSim(SPEC)
    yield s
    s.env.close()


def test_reset_obs_shapes_and_core_frame(sim):
    obs = sim.reset(CFG)
    assert obs["state"].shape == (8,) and obs["state"].dtype == np.float32
    for cam in ("agentview", "eye_in_hand"):
        assert obs[cam].shape == (256, 256, 3) and obs[cam].dtype == np.uint8 and obs[cam].std() > 0
    np.testing.assert_allclose(obs["state"][:3], sim.get_ee_pose()[:3], atol=1e-6)
    assert sim.get_ee_pose()[2] == pytest.approx(0.40, abs=0.01)


def test_objects_rest_at_their_layout_poses_after_settle(sim):
    sim.reset(CFG)
    for name, want in object_poses(SPEC, CFG).items():
        got = sim.get_object_pose(name)
        np.testing.assert_allclose(got[:2], want[:2], atol=0.005)
        assert got[2] == pytest.approx(SPEC.objects[name].center_z(), abs=0.003)
        assert abs(np.angle(np.exp(1j * (quat_yaw(got[3:]) - quat_yaw(want[3:]))))) < 0.05


def test_expert_episode_succeeds_with_all_stage_flags(sim):
    frames = []
    r = run_expert_episode(sim, SPEC, CFG, ROBOSUITE_EXPERT_PARAMS, record=lambda o, a: frames.append(a.shape))
    assert r.success, r.trace
    assert all(r.final_info.stage_flags.values())
    assert sim.tray_contains(CFG.object)
    assert len(frames) == r.steps and frames[0] == (7,)
