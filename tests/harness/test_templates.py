import json
from pathlib import Path
import pytest
import yaml
from harness.agents import load_agents
from harness.model import allowed_for, checks_for, load_templates, render_prompt, validate_params, ParamError
from orchestrator.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = load_templates(ROOT / "harness/tasks")
EXPECTED = {"add_object", "extend_dataset", "add_variation_axis", "add_sim_backend", "add_policy_adapter",
            "add_expert", "free_task"}

SAMPLE = {
    "add_object": {"name": "hoop", "label": "Hoop", "radius": 0.06, "length": 0.02, "mass": 0.05},
    "extend_dataset": {"out": "/tmp/x"},
    "add_variation_axis": {"axis_id": "camera_jitter", "label": "Camera jitter", "description": "±2 cm"},
    "add_sim_backend": {"sim": "mujoco"},
    "add_policy_adapter": {"model_id": "pi05"},
    "add_expert": {},
    "free_task": {"goal": "fix it", "paths": "eval/, tests/eval/"},
}


def test_all_starter_templates_present():
    assert set(TEMPLATES) == EXPECTED


@pytest.mark.parametrize("task_id", sorted(EXPECTED))
def test_template_renders_with_sample_params(task_id):
    t = TEMPLATES[task_id]
    params = validate_params(t, SAMPLE[task_id])
    prompt = render_prompt(t, params)
    assert "{" not in "".join(allowed_for(t, params)) and "{" not in " ".join(c.cmd for c in checks_for(t, params))
    assert t.title in prompt and allowed_for(t, params)


def test_catalog_harness_tasks_exist():
    cat = load_catalog(ROOT / "configs/catalog.yaml")
    refs = {i.harness_task for s in cat.sections.values() for i in s.items if i.harness_task}
    refs |= {s.add_task for s in cat.sections.values() if s.add_task}
    assert refs <= set(TEMPLATES)


def test_agents_config_has_copilot_default():
    agents = load_agents(ROOT / "configs/agents.yaml")
    assert agents.default == "copilot" and {"copilot", "claude", "codex"} <= set(agents.agents)
    assert "{prompt}" in agents.get(None).argv and "{workdir}" in agents.get(None).argv


def test_assert_expert_rate(tmp_path):
    from scripts.assert_expert_rate import main
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"per_object": {"a": {"rate": 1.0, "success": 5, "attempted": 5},
                                            "b": {"rate": 0.6, "success": 3, "attempted": 5}}}))
    assert main([str(p), "--min", "0.5"]) == 0
    assert main([str(p), "--min", "0.8"]) == 1


def test_identifier_patterns_reject_shell_metacharacters():
    """Patterns prevent shell injection in identifiers."""
    # Test that add_object's name param rejects shell metacharacters
    t_obj = TEMPLATES["add_object"]
    valid_params = SAMPLE["add_object"].copy()
    valid_params["name"] = "valid_name_123"
    validate_params(t_obj, valid_params)  # Should not raise
    
    # Test that invalid names are rejected
    invalid_params = SAMPLE["add_object"].copy()
    invalid_params["name"] = "x; rm -rf ~"
    with pytest.raises(ParamError, match="pattern"):
        validate_params(t_obj, invalid_params)
    
    # Test that free_task's check param stays raw (type: command)
    t_free = TEMPLATES["free_task"]
    check_params = {"goal": "g", "paths": "eval/", "check": "uv run pytest -q && echo ok"}
    validated = validate_params(t_free, check_params)
    checks = checks_for(t_free, validated)
    assert checks[0].cmd == "uv run pytest -q && echo ok"
