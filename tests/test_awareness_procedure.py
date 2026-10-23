from pathlib import Path
from types import SimpleNamespace
import pytest
from awareness.procedure import ProcedureSpec, ProcedureStep

ROOT = Path(__file__).resolve().parents[1]
PROC = ProcedureSpec.from_yaml(ROOT / "configs/procedures/coronary_angiography.yaml")
TASK = SimpleNamespace(objects=["bottle", "vial", "syringe"])


def test_steps_are_ordered_and_unique():
    ids = [s.step_id for s in PROC.steps]
    assert len(ids) == len(set(ids)) >= 5


def test_duplicate_step_ids_are_rejected():
    step = ProcedureStep(step_id="a", description="d", tools=["vial"])
    with pytest.raises(ValueError, match="duplicate"):
        ProcedureSpec(procedure_id="p", steps=[step, step])


def test_all_tools_exist_in_task():
    PROC.validate_against(TASK)


def test_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="bottle"):
        PROC.validate_against(SimpleNamespace(objects=["vial", "syringe"]))


def test_next_tool_is_first_tool_of_following_step():
    assert PROC.next_tool("access_prep") == "syringe"


def test_next_tool_skips_steps_that_need_no_tool():
    assert PROC.next_tool("sheath_insertion") == "bottle"
    assert PROC.next_tool("catheter_advance") == "bottle"


def test_next_tool_is_none_when_no_later_step_needs_a_tool():
    assert PROC.next_tool("contrast_injection") is None
    assert PROC.next_tool("closure") is None


def test_unknown_step_raises():
    with pytest.raises(KeyError):
        PROC.next_tool("nope")


def test_every_procedure_tool_has_a_trained_instruction_in_the_real_task():
    from contracts import TaskSpec, load_yaml

    task = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)
    PROC.validate_against(task)
    for tool in {t for s in PROC.steps for t in s.tools}:
        assert task.instructions(tool)[0] == f"Hand me the {tool}"
