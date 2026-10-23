from pathlib import Path
import pytest
from orchestrator.catalog import load_catalog, selection_problems

ROOT = Path(__file__).resolve().parents[2]
CAT = load_catalog(ROOT / "configs/catalog.yaml")


def test_catalog_has_all_sections_and_defaults_are_runnable():
    assert set(CAT.sections) == {"simulator", "embodiment", "objects", "expert", "variation", "model"}
    assert selection_problems(CAT, dict(CAT.defaults)) == []


def test_every_non_ready_item_points_to_a_harness_task():
    for section in CAT.sections.values():
        for item in section.items:
            if item.status != "ready":
                assert item.harness_task, item.id


def test_not_ready_item_reports_harness_task():
    sel = dict(CAT.defaults, simulator="mujoco", embodiment="fr3_duo")
    kinds = {(p.item, p.kind): p.harness_task for p in selection_problems(CAT, sel)}
    assert kinds[("mujoco", "not_ready")] == "add_engine"
    assert ("fr3_duo", "not_ready") in kinds


def test_requires_rule_blocks_panda_in_mujoco():
    sel = dict(CAT.defaults, simulator="mujoco")
    assert any(p.item == "panda" and p.kind == "requires" for p in selection_problems(CAT, sel))


def test_unknown_object_points_to_add_object():
    sel = dict(CAT.defaults, objects=["bottle", "guidewire"])
    p = next(p for p in selection_problems(CAT, sel) if p.kind == "unknown_item")
    assert p.item == "guidewire" and p.harness_task == "add_object"


@pytest.mark.parametrize("sel_patch,kind", [
    ({"objects": []}, "min"),
    ({"objects": "bottle"}, "type"),
    ({"extra": "x"}, "unknown_section"),
])
def test_shape_errors(sel_patch, kind):
    sel = dict(CAT.defaults, **sel_patch)
    assert kind in {p.kind for p in selection_problems(CAT, sel)}


def test_missing_section():
    sel = dict(CAT.defaults)
    del sel["model"]
    assert any(p.section == "model" and p.kind == "missing" for p in selection_problems(CAT, sel))
