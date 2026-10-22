from pathlib import Path
import numpy as np
from awareness.assistant import Assistant
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import ScriptedVLM, VLMReading

ROOT = Path(__file__).resolve().parents[1]
PROC = ProcedureSpec.from_yaml(ROOT / "configs/procedures/coronary_angiography.yaml")
FRAME = np.zeros((224, 224, 3), dtype=np.uint8)


class Task:
    def instructions(self, obj):
        return [f"Hand me the {obj}"]


def make(readings, confirm, **kw):
    spoken = []
    engine = AwarenessEngine(PROC, ScriptedVLM(readings))
    return Assistant(engine, Task(), confirm=confirm, speak=spoken.append, **kw), spoken


def stream(n, dt=10.0):
    return [(i * dt, FRAME) for i in range(n)]


def test_proposes_syringe_at_access_prep_and_emits_on_yes():
    a, spoken = make([VLMReading("access_prep", 0.9, "r")], confirm=lambda q: True)
    assert list(a.run(stream(1))) == ["Hand me the syringe"]
    assert "syringe" in spoken[0]


def test_proposes_contrast_bottle_during_sheath_insertion():
    a, _ = make([VLMReading("sheath_insertion", 0.9, "r")], confirm=lambda q: True)
    assert list(a.run(stream(1))) == ["Hand me the bottle"]


def test_declined_tool_is_not_reoffered_in_same_step():
    asked = []
    a, _ = make([VLMReading("access_prep", 0.9, "r")], confirm=lambda q: asked.append(q) or False)
    assert list(a.run(stream(5))) == []
    assert len(asked) == 1


def test_declined_tool_is_reoffered_after_step_changes():
    asked = []
    readings = [VLMReading("sheath_insertion", 0.9, "a"), VLMReading("catheter_advance", 0.9, "b")]
    a, _ = make(readings, confirm=lambda q: asked.append(q) or False)
    list(a.run(stream(2, dt=30.0)))
    assert len(asked) == 2


def test_accepted_tool_is_not_reoffered():
    asked = []
    a, _ = make([VLMReading("access_prep", 0.9, "r")], confirm=lambda q: asked.append(q) or True)
    assert len(list(a.run(stream(5)))) == 1
    assert len(asked) == 1


def test_cooldown_blocks_back_to_back_proposals():
    asked = []
    readings = [VLMReading("access_prep", 0.9, "a"), VLMReading("anaesthetic_injection", 0.9, "b")]
    a, _ = make(readings, confirm=lambda q: asked.append(q) or False, cooldown_s=30)
    list(a.run([(0.0, FRAME), (5.0, FRAME)]))
    assert len(asked) == 1


def test_no_proposal_when_step_unknown():
    a, _ = make([VLMReading("closure", 0.1, "r")], confirm=lambda q: True)
    assert list(a.run(stream(3))) == []


def test_no_proposal_when_no_later_step_needs_a_tool():
    a, _ = make([VLMReading("closure", 0.9, "r")], confirm=lambda q: True)
    assert list(a.run(stream(3))) == []
