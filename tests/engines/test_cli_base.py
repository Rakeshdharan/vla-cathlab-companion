import json
import sys
import pytest
from engines.cli_base import Flag, Stage, run_cli
from engines.registry import get_engine

RS = get_engine("robosuite")


def _write(args, ctx):
    (ctx.out / "ok.txt").write_text(f"n={args.n} objects={','.join(ctx.objects)}")


def _script(args, ctx):
    return ctx.run_script(["-c", "print('hi')"])


def _boom(args, ctx):
    raise RuntimeError("no simulator here")


STAGES = {
    "write": Stage(_write, flags=(Flag("--n", int, 3),)),
    "script": Stage(_script),
    "boom": Stage(_boom),
    "need": Stage(lambda a, c: None, flags=(Flag("--dataset", required=True), Flag("--param", action="append"))),
}


def test_stage_gets_flags_out_and_run_objects(tmp_path):
    run = tmp_path / "run.yaml"
    run.write_text("id: r1\nselection: {objects: [vial, syringe]}\n")
    out = tmp_path / "out"
    assert run_cli("robosuite", STAGES, ["write", "--out", str(out), "--run", str(run), "--n", "5"], manifest=RS) == 0
    assert (out / "ok.txt").read_text() == "n=5 objects=vial,syringe"


def test_dry_run_prints_argv_and_creates_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert run_cli("robosuite", STAGES, ["script", "--out", str(out), "--dry-run"], manifest=RS) == 0
    cmd = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert cmd == [sys.executable, "-c", "print('hi')"] and not out.exists()


def test_failure_is_one_line_on_stderr(tmp_path, capsys):
    assert run_cli("robosuite", STAGES, ["boom", "--out", str(tmp_path)], manifest=RS) == 1
    assert capsys.readouterr().err.strip() == "robosuite boom failed: no simulator here"


def test_script_exit_code_propagates(tmp_path, capsys):
    stages = {"bad": Stage(lambda a, c: c.run_script(["-c", "raise SystemExit(3)"]))}
    assert run_cli("robosuite", stages, ["bad", "--out", str(tmp_path)], manifest=RS) == 3
    assert "robosuite bad failed: exit code 3" in capsys.readouterr().err


def test_required_and_unknown_are_usage_errors(tmp_path):
    for argv in (["nope", "--out", str(tmp_path)], ["need", "--out", str(tmp_path)]):
        with pytest.raises(SystemExit) as e:
            run_cli("robosuite", STAGES, argv, manifest=RS)
        assert e.value.code == 2


def test_append_flag_collects_values(tmp_path):
    seen = {}
    stages = {"s": Stage(lambda a, c: seen.update(p=a.param), flags=(Flag("--param", action="append"),))}
    run_cli("robosuite", stages, ["s", "--out", str(tmp_path), "--param", "a=1", "--param", "b=2"], manifest=RS)
    assert seen["p"] == ["a=1", "b=2"]
    run_cli("robosuite", stages, ["s", "--out", str(tmp_path)], manifest=RS)
    assert seen["p"] == []
