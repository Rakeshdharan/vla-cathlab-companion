"""ManiSkill engine CLI wiring (GPU-free: every stage is checked with --dry-run, mani_skill is never imported)."""
import json
import sys
from engines.maniskill.cli import main
from engines.registry import ROOT, get_engine

SPEC = str(ROOT / "configs/variation/cathlab_v1.yaml")


def dry(capsys, *args) -> dict | list:
    assert main([*args, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_manifest():
    m = get_engine("maniskill")
    assert m.io == "panda_libero" and m.backend == "sim.maniskill.backend:ManiSkillBackend"
    assert m.entry == "engines.maniskill.cli" and m.embodiments == ["panda"]


def test_preview(tmp_path, capsys):
    out = dry(capsys, "preview", "--out", str(tmp_path), "--n", "4")
    assert out["stage"] == "preview" and out["out_png"] == str(tmp_path / "grid.png") and out["spec_path"] == SPEC


def test_check_expert_uses_run_objects_and_params(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial]}\n")
    out = dry(capsys, "check-expert", "--out", str(tmp_path), "--run", str(run), "--n", "5", "--param", "pos_tol=0.01")
    assert out["objects"] == "['vial']" and out["params"] == "['pos_tol=0.01']" and out["n"] == "5"
    assert out["json_path"] == str(tmp_path / "expert_check.json")


def test_generate_window_and_shard(tmp_path, capsys):
    out = dry(capsys, "generate", "--out", str(tmp_path), "--n", "384", "--shard", "2", "--num-shards", "8")
    assert (out["n"], out["shard"], out["num_shards"]) == ("384", "2", "8")
    assert out["dataset_id"] == "cathlab/pick_maniskill"


def test_convert_uses_engine_agnostic_script(tmp_path, capsys):
    cmd = dry(capsys, "convert", "--out", str(tmp_path), "--dataset", "data/x")
    assert cmd[:2] == [sys.executable, "scripts/npz_to_lerobot.py"] and cmd[cmd.index("--fps") + 1] == "20"


def test_eval_defaults(tmp_path, capsys):
    out = dry(capsys, "eval", "--out", str(tmp_path), "--max-seeds", "1", "--tiers", "in_distribution")
    assert out["policy_arg"] == "expert" and out["tiers"] == "['in_distribution']"
    assert out["suite_path"] == str(ROOT / "configs/eval/suite_v2.yaml")


def test_select_configs_pairs_with_robosuite():
    from dataengine.variation import VariationSpec, generation_order, train_configs
    from engines.maniskill.stages import select_configs
    spec = VariationSpec.load(SPEC)
    shards = [select_configs(spec, "train", 0, 384, i, 4) for i in range(4)]
    order = generation_order(spec, train_configs(spec))
    assert sorted((c.seed for s in shards for c in s)) == sorted(c.seed for c in order)
    assert shards[1][0] == order[1]
