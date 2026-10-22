from pathlib import Path
import numpy as np
import pytest
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import ScriptedVLM, VLMReading

ROOT = Path(__file__).resolve().parents[1]
PROC = ProcedureSpec.from_yaml(ROOT / "configs/procedures/coronary_angiography.yaml")
FRAME = np.zeros((224, 224, 3), dtype=np.uint8)


def engine(*readings, min_confidence=0.5):
    return AwarenessEngine(PROC, ScriptedVLM(list(readings)), min_confidence=min_confidence)


def test_access_prep_predicts_syringe_with_rationale():
    s = engine(VLMReading("access_prep", 0.9, "draping the site")).observe(FRAME, 2.0)
    assert (s.step, s.next_tool, s.rationale, s.t) == ("access_prep", "syringe", "draping the site", 2.0)


def test_sheath_insertion_predicts_contrast_bottle():
    s = engine(VLMReading("sheath_insertion", 0.9, "sheath entering vessel")).observe(FRAME, 0.0)
    assert (s.step, s.next_tool) == ("sheath_insertion", "bottle")


def test_low_confidence_before_any_step_gives_no_step():
    s = engine(VLMReading("closure", 0.2, "blurry")).observe(FRAME, 0.0)
    assert s.step is None and s.next_tool is None


def test_low_confidence_keeps_previous_step():
    e = engine(VLMReading("sheath_insertion", 0.9, "a"), VLMReading("closure", 0.3, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "sheath_insertion"


def test_regression_is_ignored():
    e = engine(VLMReading("catheter_advance", 0.9, "a"), VLMReading("access_prep", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "catheter_advance"


def test_skipping_ahead_is_limited_to_one_step():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("closure", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "access_prep"


def test_advancing_one_step_is_accepted():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("anaesthetic_injection", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "anaesthetic_injection"


def test_scripted_vlm_repeats_last_reading():
    vlm = ScriptedVLM([VLMReading("access_prep", 0.9, "a")])
    assert vlm.classify(FRAME, PROC) == vlm.classify(FRAME, PROC)


def test_gemini_client_never_reaches_the_network_without_a_key(monkeypatch):
    from awareness.vlm import GeminiVLM

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises((ImportError, KeyError)):
        GeminiVLM().classify(FRAME, PROC)
