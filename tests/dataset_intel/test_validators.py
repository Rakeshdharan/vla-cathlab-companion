import numpy as np
from dataengine.variation import VariationSpec, train_configs
from dataset_intel import validators as V
from dataset_intel.coverage import coverage
from dataset_intel.report import build_report, load_config, render_md
from engines.registry import ROOT

SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
CFG = load_config()


def episode(n=100, seed=0):
    rng = np.random.default_rng(seed)
    state = np.cumsum(rng.normal(0, 0.01, (n, 8)), axis=0)
    action = np.clip(rng.normal(0, 0.3, (n, 7)), -1, 1)
    action[:, 6] = np.where(np.arange(n) > n // 2, 1.0, -1.0)
    return state, action


def kinds(issues):
    return {i.kind for i in issues}


def test_clean_episode_has_no_issues():
    s, a = episode()
    issues = (V.check_nan(0, s, a) + V.check_frozen(0, s, 20, 2.0, 1e-6) + V.check_gripper(0, a, "panda_libero")
              + V.check_action_jump(0, a, "panda_libero", 1.5) + V.check_idle(0, V.idle_fraction(a, "panda_libero"), 0.2))
    assert issues == []


def test_each_validator_fires():
    s, a = episode()
    s2 = s.copy(); s2[3, 2] = np.nan
    assert kinds(V.check_nan(0, s2, a)) == {"nan"}
    s3 = s.copy(); s3[10:60] = s3[10]
    assert kinds(V.check_frozen(0, s3, 20, 2.0, 1e-6)) == {"frozen"}
    a2 = a.copy(); a2[:, 6] = -1.0
    assert kinds(V.check_gripper(0, a2, "panda_libero")) == {"gripper_never_closes"}
    a3 = a.copy(); a3[20, 0], a3[21, 0] = -1.0, 1.0
    assert kinds(V.check_action_jump(0, a3, "panda_libero", 1.5)) == {"action_jump"}
    a4 = a.copy(); a4[:50, :6] = 0.0; a4[:50, 6] = -1.0
    assert kinds(V.check_idle(0, V.idle_fraction(a4, "panda_libero"), 0.2)) == {"idle_fraction"}
    assert kinds(V.check_frame_counts(0, 100, {"observation.images.agentview": 99})) == {"frame_count_mismatch"}
    lengths = {i: 100 for i in range(10)} | {10: 400}
    assert [i.episode for i in V.length_outliers(lengths, 1.5)] == [10]


def test_fr3_gripper_convention():
    a = np.ones((50, 16)); a[:, 15] = 1.0
    assert kinds(V.check_gripper(0, a, "fr3_duo_16d")) == {"gripper_never_closes"}
    a[30:, 15] = 0.0
    assert V.check_gripper(0, a, "fr3_duo_16d") == []


def test_coverage_and_readiness():
    cfgs = [c for c in train_configs(SPEC)][:96]
    extra = [{"config": c.model_dump(), "source_sim": "robosuite"} for c in cfgs]
    cov = coverage(extra, SPEC)
    assert cov.train_cells == 384 and len(cov.empty_cells) == 384 - 96 and sum(cov.per_source_sim.values()) == 96
    eps = {i: dict(zip(("state", "action"), episode(seed=i))) for i in range(4)}
    eps[3]["action"][:, 6] = -1.0                                  # one error episode
    r = build_report({"fps": 20, "robot_type": "panda"}, eps, {}, "panda_libero", CFG, extra, SPEC, root="x")
    assert r.readiness.valid_fraction == 0.75 and r.readiness.coverage_fraction == 0.25
    assert 0 < r.readiness.score < 1 and "gripper_never_closes (error)" in render_md(r)


def test_fr3_gripper_switches_are_not_action_jumps():
    a = np.zeros((40, 16)); a[20:, 7] = 1.0; a[25:, 15] = 1.0     # left/right gripper flips
    assert V.check_action_jump(0, a, "fr3_duo_16d", 0.25) == []
    a[30, 3] = 0.5
    assert kinds(V.check_action_jump(0, a, "fr3_duo_16d", 0.25)) == {"action_jump"}
