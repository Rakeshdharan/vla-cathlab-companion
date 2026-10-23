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


def test_advancing_needs_the_new_step_twice_in_a_row():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("anaesthetic_injection", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "access_prep"
    assert e.observe(FRAME, 2.0).step == "anaesthetic_injection"


def test_one_noisy_forward_reading_does_not_move_the_step():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("anaesthetic_injection", 0.9, "b"),
               VLMReading("access_prep", 0.9, "c"), VLMReading("access_prep", 0.9, "d"))
    assert [e.observe(FRAME, float(t)).step for t in range(4)] == ["access_prep"] * 4


def test_scripted_vlm_repeats_last_reading():
    vlm = ScriptedVLM([VLMReading("access_prep", 0.9, "a")])
    assert vlm.classify(FRAME, PROC) == vlm.classify(FRAME, PROC)


def test_gemini_client_never_reaches_the_network_without_a_key(monkeypatch):
    from awareness.vlm import GeminiVLM

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises((ImportError, KeyError)):
        GeminiVLM().classify(FRAME, PROC)


def test_engine_hands_the_vlm_a_sliding_window_of_recent_frames():
    seen = []

    class Spy:
        def classify(self, frames, procedure):
            seen.append(len(frames))
            return VLMReading("access_prep", 0.9, "r")

    e = AwarenessEngine(PROC, Spy(), window=3)
    for t in range(5):
        e.observe(FRAME, float(t))
    assert seen == [1, 2, 3, 3, 3]


def test_unknown_step_from_vlm_is_ignored():
    s = engine(VLMReading("bogus", 0.99, "r")).observe(FRAME, 0.0)
    assert s.step is None


def test_matching_tool_raises_confidence_but_not_the_proposal():
    s = engine(VLMReading("access_prep", 0.7, "r", tool="vial")).observe(FRAME, 0.0)
    assert s.corroborated and s.observed_tool == "vial" and s.confidence > 0.7 and s.next_tool == "syringe"


def test_off_step_tool_never_changes_next_tool_or_lowers_confidence():
    s = engine(VLMReading("access_prep", 0.7, "r", tool="bottle")).observe(FRAME, 0.0)
    assert not s.corroborated and s.confidence == 0.7 and s.next_tool == "syringe"


def test_min_interval_skips_the_vlm_between_queries():
    calls = []

    class Spy:
        def classify(self, frames, procedure):
            calls.append(1)
            return VLMReading("access_prep", 0.9, "r")

    e = AwarenessEngine(PROC, Spy(), min_interval_s=5.0)
    states = [e.observe(FRAME, t) for t in (0.0, 1.0, 4.9, 5.0)]
    assert len(calls) == 2 and states[1].step == "access_prep"


def test_corroborating_tool_can_lift_a_reading_over_the_threshold():
    assert engine(VLMReading("access_prep", 0.55, "r", tool="vial"), min_confidence=0.6).observe(FRAME, 0.0).step == "access_prep"
    assert engine(VLMReading("access_prep", 0.55, "r", tool="bottle"), min_confidence=0.6).observe(FRAME, 0.0).step is None
    assert engine(VLMReading("access_prep", 0.55, "r"), min_confidence=0.6).observe(FRAME, 0.0).step is None


def test_same_far_step_twice_in_a_row_is_accepted_as_a_jump():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("contrast_injection", 0.9, "b"),
               VLMReading("contrast_injection", 0.9, "c"))
    assert e.observe(FRAME, 0.0).step == "access_prep"
    assert e.observe(FRAME, 1.0).step == "access_prep"
    assert e.observe(FRAME, 2.0).step == "contrast_injection"


def test_far_step_that_is_interrupted_is_not_accepted():
    e = engine(VLMReading("access_prep", 0.9, "a"), VLMReading("contrast_injection", 0.9, "b"),
               VLMReading("access_prep", 0.9, "c"), VLMReading("contrast_injection", 0.9, "d"))
    assert [e.observe(FRAME, float(t)).step for t in range(4)] == ["access_prep"] * 4


def test_a_jump_never_goes_backwards():
    e = engine(VLMReading("catheter_advance", 0.9, "a"), VLMReading("access_prep", 0.9, "b"),
               VLMReading("access_prep", 0.9, "c"))
    assert [e.observe(FRAME, float(t)).step for t in range(3)] == ["catheter_advance"] * 3
