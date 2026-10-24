from scripts.demo_bootstrap import bootstrap, eval_paths


def test_bootstrap_rebuilds_the_h37_registry_and_is_idempotent(tmp_path):
    assert set(eval_paths("smolvla_panda_pick_v1")) == {"robosuite", "maniskill", "genesis"}
    first = bootstrap(tmp_path)
    assert first["production"] == "v2"
    assert any(line.startswith("promote v2 (smolvla_panda_pick_v1): yes") for line in first["log"])
    assert any(line.startswith("promote v6 (xvla_panda_genesis_v1): no") for line in first["log"])
    assert (tmp_path / "reports" / "rep-cathlab-pick-v2" / "next_run.json").is_file()
    again = bootstrap(tmp_path)
    assert again["production"] == "v2" and not any(line.startswith("register") for line in again["log"])
