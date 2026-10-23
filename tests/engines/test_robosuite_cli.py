import json
import sys
import pytest
from engines.registry import ROOT
from engines.robosuite.cli import main

SPEC = str(ROOT / "configs/variation/cathlab_v1.yaml")


def argv_of(capsys, *args):
    assert main([*args, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def val(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_preview(tmp_path, capsys):
    cmd = argv_of(capsys, "preview", "--out", str(tmp_path), "--n", "4")
    assert cmd[:2] == [sys.executable, "scripts/robosuite_variation_grid.py"]
    assert val(cmd, "--out") == str(tmp_path / "grid.png") and val(cmd, "--n") == "4" and val(cmd, "--spec") == SPEC


def test_check_expert_uses_run_objects_and_params(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial]}\n")
    cmd = argv_of(capsys, "check-expert", "--out", str(tmp_path), "--run", str(run), "--n", "5",
                  "--param", "pos_tol=0.01")
    assert cmd[1] == "scripts/robosuite_expert_check.py"
    assert val(cmd, "--objects") == "vial" and val(cmd, "--param") == "pos_tol=0.01" and val(cmd, "--n") == "5"
    assert val(cmd, "--json") == str(tmp_path / "expert_check.json")


def test_check_expert_without_run_checks_all_objects(tmp_path, capsys):
    cmd = argv_of(capsys, "check-expert", "--out", str(tmp_path))
    assert "--objects" not in cmd and val(cmd, "--n") == "20"


def test_generate_window_and_shard(tmp_path, capsys):
    cmd = argv_of(capsys, "generate", "--out", str(tmp_path), "--start", "10", "--n", "6", "--shard", "1",
                  "--num-shards", "2")
    assert cmd[1] == "scripts/robosuite_generate.py"
    expected = {"--start": "10", "--n": "6", "--shard": "1", "--num-shards": "2", "--out": str(tmp_path),
                "--version": "v1", "--dataset-id": "cathlab/pick_robosuite", "--split": "train", "--spec": SPEC,
                "--task": str(ROOT / "configs/tasks/cathlab_pick.yaml")}
    assert {k: val(cmd, k) for k in expected} == expected


def test_convert(tmp_path, capsys):
    cmd = argv_of(capsys, "convert", "--out", str(tmp_path), "--dataset", "var/runs/r1/generate/j1")
    assert cmd[1] == "scripts/npz_to_lerobot.py"
    assert val(cmd, "--dataset") == "var/runs/r1/generate/j1" and val(cmd, "--root") == str(tmp_path / "lerobot")
    assert val(cmd, "--fps") == "20" and "--repo-id" not in cmd
    with pytest.raises(SystemExit):
        main(["convert", "--out", str(tmp_path), "--dry-run"])          # --dataset is required


def test_eval_and_rollout_dry_run(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial, bottle]}\n")
    ev = argv_of(capsys, "eval", "--out", str(tmp_path), "--run", str(run), "--tiers", "in_distribution",
                 "--max-seeds", "1")
    assert ev["stage"] == "eval" and ev["policy_arg"] == "expert" and ev["objects"] == "['vial', 'bottle']"
    assert ev["tiers"] == "['in_distribution']" and ev["max_seeds"] == "1"
    assert ev["suite_path"] == str(ROOT / "configs/eval/suite_v2.yaml")
    ro = argv_of(capsys, "rollout", "--out", str(tmp_path), "--run", str(run), "--instruction", "Pass the vial")
    assert ro["obj"] == "vial" and ro["seed"] == "5000" and ro["instruction"] == "Pass the vial"


def test_help_works_without_robosuite(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "check-expert" in capsys.readouterr().out
