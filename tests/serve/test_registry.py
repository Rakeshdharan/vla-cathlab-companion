import json
import pytest
import yaml
from contracts import EpisodeResult, EvalReport, ModelCard
from serve.registry import MODEL_NAME, PromotionRule, Registry, main, policy_string


def report(tmp_path, name, successes, n=60, p95=100.0, engine="robosuite"):
    eps = [EpisodeResult(episode=i, seed=5000 + i % 4, object="vial", instruction="Hand me the vial",
                         success=i < successes, failed_stage=None if i < successes else "grasp", steps=100,
                         latency_ms_p50=p95 / 2, latency_ms_p95=p95, scene_variant="in_distribution")
           for i in range(n)]
    path = tmp_path / f"{name}_{engine}.json"
    path.write_text(EvalReport(model_version=name, dataset_versions=[], backend=engine, episodes=eps).model_dump_json())
    return str(path)


def card(version, ckpt="user/smolvla_panda_pick_v1", adapter="lerobot"):
    return ModelCard(model_name=MODEL_NAME, version=version, adapter=adapter, ckpt_uri=ckpt,
                     dataset_versions=["Rakeshdharan56/pick_robosuite@v1"], eval_report_path="")


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "registry", repo_root=tmp_path)


def test_register_list_get_and_duplicates(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 30)})
    assert [m.card.version for m in reg.list_versions(MODEL_NAME)] == ["v1", "v2"]
    assert reg.get(MODEL_NAME, "v2").card.eval_report_path.endswith("b_robosuite.json")
    with pytest.raises(FileExistsError):
        reg.register(card("v1"), {"robosuite": report(tmp_path, "c", 1)})
    with pytest.raises(KeyError):
        reg.get(MODEL_NAME, "v9")
    with pytest.raises(FileNotFoundError):
        reg.register(card("v3"), {"robosuite": str(tmp_path / "missing.json")})


def test_first_promotion_needs_only_latency_and_episodes(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20, p95=271.0)})
    d = reg.promote(MODEL_NAME, "v1")
    assert d.promoted and d.baseline_version is None and d.p95_ms == pytest.approx(271.0)
    assert reg.production(MODEL_NAME).card.version == "v1" and reg.get(MODEL_NAME, "v1").card.promoted


def test_candidate_must_beat_production_beyond_ci(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})       # 33 %
    reg.promote(MODEL_NAME, "v1")
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 24)})       # 40 %, CI low ~28 % < 33 %
    d = reg.promote(MODEL_NAME, "v2")
    assert not d.promoted and "does not beat" in d.reason and d.baseline_version == "v1"
    reg.register(card("v3"), {"robosuite": report(tmp_path, "c", 45)})       # 75 %, CI low ~63 % > 33 %
    assert reg.promote(MODEL_NAME, "v3").promoted
    assert reg.production(MODEL_NAME).card.version == "v3"
    assert not reg.get(MODEL_NAME, "v1").card.promoted and reg.get(MODEL_NAME, "v3").card.promoted


def test_latency_episodes_engine_and_force(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 50, p95=900.0)})
    d = reg.promote(MODEL_NAME, "v1")
    assert not d.promoted and "latency" in d.reason
    assert not reg.promote(MODEL_NAME, "v1", force=True).promoted          # force never skips latency
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 5, n=10)})
    assert "insufficient eval" in reg.promote(MODEL_NAME, "v2").reason
    reg.register(card("v3"), {"maniskill": report(tmp_path, "c", 50, engine="maniskill")})
    assert "no eval on robosuite" in reg.promote(MODEL_NAME, "v3").reason
    reg.register(card("v4"), {"robosuite": report(tmp_path, "d", 50)})
    reg.promote(MODEL_NAME, "v4")
    reg.register(card("v5"), {"robosuite": report(tmp_path, "e", 1)})
    d = reg.promote(MODEL_NAME, "v5", force=True)
    assert d.promoted and d.reason == "forced by user"


def test_rollback_and_history(reg, tmp_path):
    for v, s in (("v1", 20), ("v2", 45)):
        reg.register(card(v), {"robosuite": report(tmp_path, v, s)})
        reg.promote(MODEL_NAME, v)
    assert reg.rollback(MODEL_NAME).card.version == "v1"
    assert reg.production(MODEL_NAME).card.version == "v1" and not reg.get(MODEL_NAME, "v2").card.promoted
    assert [h["action"] for h in reg.history(MODEL_NAME)] == ["promote", "promote", "rollback"]
    assert reg.rollback(MODEL_NAME) is None and reg.production(MODEL_NAME) is None


def test_repromoting_production_is_a_noop(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})
    d1 = reg.promote(MODEL_NAME, "v1")
    assert d1.promoted
    d2 = reg.promote(MODEL_NAME, "v1")
    assert not d2.promoted and "already production" in d2.reason
    assert [h["action"] for h in reg.history(MODEL_NAME)] == ["promote"]
    assert reg.rollback(MODEL_NAME) is None
    assert reg.production(MODEL_NAME) is None


def test_concurrent_register_and_promote_is_consistent(reg, tmp_path):
    import threading

    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            v = f"v{i + 1}"
            reg.register(card(v), {"robosuite": report(tmp_path, v, 20 + i)})
            reg.promote(MODEL_NAME, v, force=True)
        except Exception as e:  # pragma: no cover - surfaced via errors list
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    history = reg.history(MODEL_NAME)
    promotions = [h for h in history if h["action"] == "promote"]
    assert len(history) == len(promotions) == 10
    for m in reg.list_versions(MODEL_NAME):
        p = reg._path(MODEL_NAME, m.card.version)
        assert yaml.safe_load(p.read_text()) is not None
    prod_path = reg._dir(MODEL_NAME) / "production.yaml"
    assert yaml.safe_load(prod_path.read_text()) is not None


def test_resolve_and_policy_string(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})
    reg.register(card("v2", ckpt="", adapter="expert"), {"robosuite": report(tmp_path, "b", 60)})
    reg.promote(MODEL_NAME, "v1")
    assert reg.policy_for(f"{MODEL_NAME}@production") == "lerobot:user/smolvla_panda_pick_v1"
    assert reg.policy_for(f"{MODEL_NAME}@v2") == "expert" == policy_string(reg.get(MODEL_NAME, "v2").card)
    with pytest.raises(KeyError):
        reg.resolve("other@production")
    with pytest.raises(ValueError):
        reg.resolve("no-at-sign")


def test_rule_from_serve_yaml():
    rule = PromotionRule.from_yaml("configs/serve.yaml")
    assert rule.latency_budget_ms_p95 == 500.0 and rule.primary_engine == "robosuite" and rule.min_episodes == 30


def test_cli_register_promote_resolve(tmp_path, capsys):
    card_path = tmp_path / "card.yaml"
    card_path.write_text(f"model_name: {MODEL_NAME}\nversion: v1\nadapter: lerobot\nckpt_uri: user/m\n"
                         "dataset_versions: [d@v1]\neval_report_path: ''\n")
    root = ["--root", str(tmp_path / "reg")]
    assert main(root + ["register", "--card", str(card_path), "--eval", f"robosuite={report(tmp_path, 'a', 20)}"]) == 0
    assert main(root + ["promote", MODEL_NAME, "v1"]) == 0
    capsys.readouterr()
    assert main(root + ["resolve", f"{MODEL_NAME}@production"]) == 0
    assert json.loads(capsys.readouterr().out)["policy"] == "lerobot:user/m"
