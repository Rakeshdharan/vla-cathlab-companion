import pytest

from contracts import ModelCard
from scripts.demo_bootstrap import bootstrap, eval_paths
from serve.registry import MODEL_NAME, Registry


def test_bootstrap_rebuilds_the_h37_registry_and_is_idempotent(tmp_path):
    assert set(eval_paths("smolvla_panda_pick_v1")) == {"robosuite", "maniskill", "genesis"}
    first = bootstrap(tmp_path)
    assert first["production"] == "v2"
    assert any(line.startswith("promote v2 (smolvla_panda_pick_v1): yes") for line in first["log"])
    assert any(line.startswith("promote v6 (xvla_panda_genesis_v1): no") for line in first["log"])
    assert (tmp_path / "reports" / "rep-cathlab-pick-v2" / "next_run.json").is_file()
    again = bootstrap(tmp_path)
    assert again["production"] == "v2" and not any(line.startswith("register") for line in again["log"])


def test_bootstrap_refuses_a_foreign_registry_without_reset(tmp_path):
    reg = Registry(tmp_path / "registry")
    reg.register(ModelCard(model_name=MODEL_NAME, version="v1", adapter="lerobot",
                            ckpt_uri="Rakeshdharan56/some_manual_test_ckpt",
                            dataset_versions=["manual"], eval_report_path=""), {})
    with pytest.raises(SystemExit, match="registry has entries not created by demo-bootstrap"):
        bootstrap(tmp_path)


def test_bootstrap_reset_rebuilds_a_clean_v2_production_over_a_foreign_registry(tmp_path):
    reg = Registry(tmp_path / "registry")
    reg.register(ModelCard(model_name=MODEL_NAME, version="v1", adapter="lerobot",
                            ckpt_uri="Rakeshdharan56/some_manual_test_ckpt",
                            dataset_versions=["manual"], eval_report_path=""), {})
    result = bootstrap(tmp_path, reset=True)
    assert result["production"] == "v2"
