import sys
import numpy as np
import pytest
from dataengine.fake_sim import FakeExpertSim
from dataengine.variation import TIERS, VariationSpec, eval_configs
from engines.expert_policy import ExpertPolicy
from engines.policies import make_policy
from engines.registry import ROOT

SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


def test_random_policy_is_one_step_chunks():
    a = make_policy("random", action_dim=7).predict({}, "Hand me the vial")
    assert a.shape == (1, 7) and np.all(np.abs(a) <= 1)


def test_url_policy_without_serve_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "serve.policy_client", None)   # import -> ImportError
    with pytest.raises(RuntimeError, match="policy server client not available"):
        make_policy("url:http://127.0.0.1:8720", action_dim=7)


def test_unknown_policy():
    with pytest.raises(ValueError, match="unknown --policy"):
        make_policy("smolvla", action_dim=7)


@pytest.mark.parametrize("cfg", eval_configs(SPEC, "in_distribution", [5000]), ids=lambda c: c.object)
def test_expert_policy_succeeds_in_fake_sim(cfg):
    sim = FakeExpertSim(SPEC)
    policy = ExpertPolicy(sim, SPEC)
    obs = sim.reset(cfg)
    policy.begin(cfg)
    info = None
    for _ in range(400):
        action = policy.predict(obs, "")
        assert action.shape == (1, 7) and action.dtype == np.float32
        obs, info = sim.step(action[0])
        if info.success:
            break
    assert info.success


def test_expert_policy_requires_begin():
    with pytest.raises(RuntimeError, match="begin"):
        ExpertPolicy(FakeExpertSim(SPEC), SPEC).predict({}, "")


def test_suite_v2_is_valid():
    from engines.robosuite.evaluate import load_suite
    suite = load_suite(ROOT / "configs/eval/suite_v2.yaml")
    assert suite["tiers"] == list(TIERS) and suite["objects"] == ["bottle", "vial", "syringe"]
    assert len(suite["seeds"]) == 4 and min(suite["seeds"]) >= 5000


def test_lerobot_policy_scheme_loads_the_checkpoint(monkeypatch):
    import train.lerobot_adapter as la
    loaded = []
    monkeypatch.setattr(la.LeRobotPolicyAdapter, "load", lambda self, ckpt: loaded.append(ckpt))
    policy = make_policy("lerobot:Rakeshdharan56/act_panda_pick_v1", action_dim=7)
    assert isinstance(policy, la.LeRobotPolicyAdapter) and loaded == ["Rakeshdharan56/act_panda_pick_v1"]
