"""Tests for eval/judge.py VLM success judge."""
from eval.judge import DummyJudge, Verdict, agreement, get_judge


def test_dummy_judge_success():
    j = DummyJudge()
    v = j.judge(None, "hand me the guidewire", ground_truth_success=True)
    assert isinstance(v, Verdict)
    assert v.success is True
    assert v.failed_stage is None


def test_dummy_judge_failure():
    j = DummyJudge()
    v = j.judge(None, "hand me the guidewire", ground_truth_success=False,
                ground_truth_failed_stage="grasp")
    assert v.success is False
    assert v.failed_stage == "grasp"
    assert "grasp" in v.explanation


def test_dummy_judge_default_success_when_no_gt():
    j = DummyJudge()
    v = j.judge(None, "hand me the sheath")
    assert v.success is True


def test_verdict_to_dict():
    v = Verdict(success=True, failed_stage=None, explanation="OK", model="dummy")
    d = v.to_dict()
    assert d["success"] is True and d["failed_stage"] is None
    assert d["explanation"] == "OK"


def test_agreement_perfect():
    j = DummyJudge()
    verdicts = [
        j.judge(None, "x", ground_truth_success=True),
        j.judge(None, "x", ground_truth_success=False, ground_truth_failed_stage="grasp"),
    ]
    gts = [True, False]
    assert agreement(verdicts, gts) == 1.0


def test_agreement_partial():
    j = DummyJudge()
    # Both verdicts say success=True (second should be failure)
    verdicts = [
        j.judge(None, "x", ground_truth_success=True),
        j.judge(None, "x", ground_truth_success=True),
    ]
    gts = [True, False]
    assert agreement(verdicts, gts) == 0.5


def test_agreement_empty():
    assert agreement([], []) == 0.0


def test_get_judge_returns_dummy_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    j = get_judge(use_vlm=True)
    assert isinstance(j, DummyJudge)


def test_get_judge_no_vlm_flag():
    j = get_judge(use_vlm=False)
    assert isinstance(j, DummyJudge)


def test_agreement_rejects_mismatched_lengths():
    import pytest
    v = DummyJudge().judge(None, "hand me the vial", ground_truth_success=True)
    with pytest.raises(ValueError):
        agreement([v, v], [True])
