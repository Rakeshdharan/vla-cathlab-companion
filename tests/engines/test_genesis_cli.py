import json
import sys
import pytest
from engines.genesis.cli import main
from engines.registry import ROOT, get_engine

SPEC = str(ROOT / "configs/variation/cathlab_v1.yaml")


def argv_of(capsys, *args):
    assert main([*args, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def val(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_manifest():
    m = get_engine("genesis")
    assert m.env.kind == "venv" and m.worker == "genesis" and m.io == "panda_libero"
    assert m.entry == "engines.genesis.cli" and m.backend == "sim.genesis.backend:GenesisBackend"
    assert m.status("expert.analytic") == "ready" and m.status("convert") == "in_progress"
    assert {c.name for c in m.credits} >= {"Genesis", "MuJoCo Menagerie Franka Panda"}


def test_preview(tmp_path, capsys):
    cmd = argv_of(capsys, "preview", "--out", str(tmp_path), "--n", "4")
    assert cmd[:3] == [sys.executable, "-m", "engines.genesis.variation_grid"]
    assert val(cmd, "--out") == str(tmp_path / "grid.png") and val(cmd, "--n") == "4" and val(cmd, "--spec") == SPEC


def test_check_expert_uses_run_objects_and_params(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial]}\n")
    cmd = argv_of(capsys, "check-expert", "--out", str(tmp_path), "--run", str(run), "--n", "5",
                  "--param", "pos_tol=0.01")
    assert cmd[1:3] == ["-m", "engines.genesis.expert_check"]
    assert val(cmd, "--objects") == "vial" and val(cmd, "--param") == "pos_tol=0.01" and val(cmd, "--n") == "5"
    assert val(cmd, "--json") == str(tmp_path / "expert_check.json")


def test_generate_window_and_shard(tmp_path, capsys):
    cmd = argv_of(capsys, "generate", "--out", str(tmp_path), "--start", "10", "--n", "6", "--shard", "1",
                  "--num-shards", "2")
    assert cmd[1:3] == ["-m", "engines.genesis.generate"]
    expected = {"--start": "10", "--n": "6", "--shard": "1", "--num-shards": "2", "--out": str(tmp_path),
                "--version": "v1", "--dataset-id": "cathlab/pick_genesis", "--split": "train", "--spec": SPEC,
                "--task": str(ROOT / "configs/tasks/cathlab_pick.yaml")}
    assert {k: val(cmd, k) for k in expected} == expected
    assert "--no-genesis-axes" not in cmd
    cmd = argv_of(capsys, "generate", "--out", str(tmp_path), "--no-genesis-axes")
    assert "--no-genesis-axes" in cmd


def test_preview_can_disable_genesis_axes(tmp_path, capsys):
    assert "--no-genesis-axes" not in argv_of(capsys, "preview", "--out", str(tmp_path))
    assert "--no-genesis-axes" in argv_of(capsys, "preview", "--out", str(tmp_path), "--no-genesis-axes")


def test_add_provenance_records_per_episode_genesis_axes(tmp_path):
    from dataclasses import asdict
    from dataengine.variation import VariationSpec, train_configs
    from engines.genesis.generate import add_provenance
    from sim.genesis.randomize import genesis_axes
    cfgs = train_configs(VariationSpec.load(SPEC))[:2]
    meta = tmp_path / "episodes_extra.jsonl"
    meta.write_text("".join(json.dumps({"episode_index": i, "config": c.model_dump()}) + "\n"
                            for i, c in enumerate(cfgs)))
    add_provenance(meta, {"engine": "genesis"}, axes_on=True)
    lines = [json.loads(line) for line in meta.read_text().splitlines()]
    assert [m["provenance"] for m in lines] == [{"engine": "genesis"}] * 2
    assert [m["genesis_axes"] for m in lines] == [json.loads(json.dumps(asdict(genesis_axes(c)))) for c in cfgs]
    add_provenance(meta, {"engine": "genesis"}, axes_on=False)
    assert all(json.loads(line)["genesis_axes"] is None for line in meta.read_text().splitlines())


def test_generate_selects_the_robosuite_window():
    from dataengine.variation import VariationSpec, generation_order, train_configs
    from engines.genesis.generate import select_configs
    spec = VariationSpec.load(SPEC)
    window = generation_order(spec, train_configs(spec))[10:16]
    assert select_configs(spec, "train", 10, 6, 1, 2) == window[1::2]


def test_convert_reuses_the_shared_converter(tmp_path, capsys):
    cmd = argv_of(capsys, "convert", "--out", str(tmp_path), "--dataset", "var/runs/r1/generate/j1")
    assert cmd[1] == "scripts/npz_to_lerobot.py"
    assert val(cmd, "--root") == str(tmp_path / "lerobot") and val(cmd, "--fps") == "20"


def test_eval_and_rollout_dry_run(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial, bottle]}\n")
    ev = argv_of(capsys, "eval", "--out", str(tmp_path), "--run", str(run), "--tiers", "in_distribution",
                 "--max-seeds", "1")
    assert ev["stage"] == "eval" and ev["policy_arg"] == "expert" and ev["objects"] == "['vial', 'bottle']"
    ro = argv_of(capsys, "rollout", "--out", str(tmp_path), "--run", str(run))
    assert ro["obj"] == "vial" and ro["seed"] == "5000"


def test_grid_is_n_train_plus_one_held_out():
    from dataengine.variation import VariationSpec
    from engines.genesis.variation_grid import grid_configs
    cfgs = grid_configs(VariationSpec.load(SPEC), 8)
    assert [c.tier for c in cfgs] == ["train"] * 8 + ["all_held_out"]
    assert len({(c.obj_color_id, c.table_color_id, c.layout_id) for c in cfgs}) == 9


def test_help_works_without_genesis(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "check-expert" in capsys.readouterr().out
