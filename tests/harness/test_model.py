import pytest
from harness.guard import path_violations
from harness.model import Check, Param, ParamError, TaskTemplate, allowed_for, checks_for, render_prompt, validate_params

T = TaskTemplate(
    id="t", title="Add object", description="d",
    params=[Param(name="name"), Param(name="mass", type="float", default=0.1),
            Param(name="grasp", type="choice", choices=["top", "side"], default="top"),
            Param(name="note", type="text", required=False), Param(name="push", type="bool", default=False)],
    context=["AGENTS.md"], prompt="Add {name} ({mass} kg, {grasp} grasp).",
    allowed_paths=["configs/", "sim/robosuite/*.py"], acceptance=[Check(name="t", cmd="pytest -q")])


def test_validate_params_coerces_and_defaults():
    out = validate_params(T, {"name": "hoop", "mass": "0.05", "push": "yes"})
    assert out == {"name": "hoop", "mass": 0.05, "grasp": "top", "note": "", "push": True}


@pytest.mark.parametrize("params,msg", [
    ({}, "missing required"),
    ({"name": "x", "grasp": "pinch"}, "not in"),
    ({"name": "x", "bogus": 1}, "unknown"),
    ({"name": "x", "mass": "heavy"}, "mass"),
])
def test_validate_params_errors(params, msg):
    with pytest.raises(ParamError, match=msg):
        validate_params(T, params)


def test_render_prompt_contains_rules_params_paths_checks_and_extra():
    p = render_prompt(T, validate_params(T, {"name": "hoop"}), extra="Use blue.")
    assert "Add hoop (0.1 kg, top grasp)." in p
    assert "Do NOT run `git commit`" in p and "- AGENTS.md" in p
    assert "- sim/robosuite/*.py" in p and "`pytest -q`" in p and "Use blue." in p


def test_path_guard():
    allowed = ["configs/", "sim/robosuite/*.py", "tests/sim/test_*.py"]
    files = ["configs/a.yaml", "sim/robosuite/env.py", "tests/sim/test_x.py", "contracts/specs.py", "configsX/a"]
    assert path_violations(files, allowed) == ["contracts/specs.py", "configsX/a"]


def test_placeholders_in_allowed_paths_and_checks():
    t = TaskTemplate(id="f", title="Free", description="d", params=[Param(name="paths"), Param(name="obj")],
                     prompt="x", allowed_paths=["{paths}", "docs/EXTENDING.md"],
                     acceptance=[Check(name="c", cmd="check --objects {obj}")])
    params = {"paths": "sim/robosuite/, tests/sim/", "obj": "hoop"}
    assert allowed_for(t, params) == ["sim/robosuite/", "tests/sim/", "docs/EXTENDING.md"]
    assert checks_for(t, params)[0].cmd == "check --objects hoop"


def test_checks_for_shell_quotes_params_but_not_command_type():
    t = TaskTemplate(id="f", title="Free", description="d", params=[Param(name="name")],
                     prompt="x", allowed_paths=["docs/"],
                     acceptance=[Check(name="c", cmd="check --objects {name}")])
    params = {"name": "x; rm -rf ~"}
    assert checks_for(t, params)[0].cmd == "check --objects 'x; rm -rf ~'"

    t2 = TaskTemplate(id="g", title="Free", description="d",
                      params=[Param(name="check", type="command")],
                      prompt="x", allowed_paths=["docs/"],
                      acceptance=[Check(name="c", cmd="{check}")])
    params2 = {"check": "pytest -q && echo ok"}
    assert checks_for(t2, params2)[0].cmd == "pytest -q && echo ok"


def test_validate_params_enforces_pattern():
    t = TaskTemplate(id="h", title="Pattern", description="d",
                     params=[Param(name="name", pattern="^[a-z][a-z0-9_]*$")],
                     prompt="x", allowed_paths=["docs/"])
    with pytest.raises(ParamError, match="name"):
        validate_params(t, {"name": "Bad Name"})
    assert validate_params(t, {"name": "hoop_2"}) == {"name": "hoop_2"}


def test_path_guard_rejects_absolute_and_dotdot_paths():
    allowed = ["configs/", "*"]
    files = ["/etc/passwd", "configs/../contracts/x.py", "..\\x", "C:\\x", "configs/ok.yaml"]
    assert path_violations(files, allowed) == ["/etc/passwd", "configs/../contracts/x.py", "..\\x", "C:\\x"]


def test_frozen_paths_are_rejected_even_when_allowed():
    from harness.guard import FROZEN_PATHS, frozen_violations
    assert "configs/variation/cathlab_v1.yaml" in FROZEN_PATHS
    files = ["configs/variation/cathlab_v1.yaml", "configs/variation/cathlab_v2.yaml"]
    assert path_violations(files, ["configs/variation/", "*"]) == ["configs/variation/cathlab_v1.yaml"]
    assert frozen_violations(files) == ["configs/variation/cathlab_v1.yaml"]
