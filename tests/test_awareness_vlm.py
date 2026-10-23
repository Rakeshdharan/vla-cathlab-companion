import json
from pathlib import Path
import numpy as np
import pytest
from awareness.procedure import ProcedureSpec
from awareness.vlm import AnthropicVLM, BudgetExceeded, VLMBudget, parse_reading

ROOT = Path(__file__).resolve().parents[1]
PROC = ProcedureSpec.from_yaml(ROOT / "configs/procedures/coronary_angiography.yaml")
FRAME = np.zeros((224, 224, 3), dtype=np.uint8)
GOOD = json.dumps({"step_id": "access_prep", "confidence": 0.8, "tool": "vial", "rationale": "reaching for vial"})


def test_parse_reading_accepts_json_and_code_fences():
    r = parse_reading("```json\n" + GOOD + "\n```", PROC)
    assert (r.step_id, r.confidence, r.tool, r.rationale) == ("access_prep", 0.8, "vial", "reaching for vial")


def test_parse_reading_drops_unknown_tool_and_clamps_confidence():
    r = parse_reading(json.dumps({"step_id": "closure", "confidence": 7, "tool": "scalpel", "rationale": "x"}), PROC)
    assert r.tool is None and r.confidence == 1.0


def test_parse_reading_rejects_unknown_step():
    with pytest.raises(ValueError, match="step_id"):
        parse_reading(json.dumps({"step_id": "nope", "confidence": 0.5, "rationale": "x"}), PROC)


class FakeAnthropic:
    def __init__(self, text=GOOD, tokens=(300, 40)):
        self.calls = []
        self.text, self.tokens = text, tokens
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)

        class U:
            input_tokens, output_tokens = self.tokens

        class R:
            content = [type("B", (), {"text": self.text})()]
            usage = U

        return R


def vlm(client, **kw):
    kw.setdefault("budget", VLMBudget(max_calls=5, max_tokens=10_000))
    return AnthropicVLM(client=client, model="m", **kw)


def test_sends_at_most_max_frames_downscaled_and_caps_output_tokens():
    pytest.importorskip("PIL")
    c = FakeAnthropic()
    big = np.zeros((900, 900, 3), dtype=np.uint8)
    vlm(c, max_frames=3, max_side=64).classify([big] * 6, PROC)
    kw = c.calls[0]
    images = [b for b in kw["messages"][0]["content"] if b["type"] == "image"]
    assert len(images) == 3 and kw["max_tokens"] <= 300 and kw["thinking"] == {"type": "disabled"}


def test_call_budget_stops_before_the_api_is_hit():
    pytest.importorskip("PIL")
    c = FakeAnthropic()
    v = vlm(c, budget=VLMBudget(max_calls=2, max_tokens=10_000))
    v.classify([FRAME], PROC)
    v.classify([FRAME + 1], PROC)
    with pytest.raises(BudgetExceeded):
        v.classify([FRAME + 2], PROC)
    assert len(c.calls) == 2


def test_token_budget_uses_reported_usage():
    pytest.importorskip("PIL")
    c = FakeAnthropic(tokens=(900, 200))
    v = vlm(c, budget=VLMBudget(max_calls=50, max_tokens=1000))
    v.classify([FRAME], PROC)
    with pytest.raises(BudgetExceeded):
        v.classify([FRAME + 1], PROC)


def test_identical_frames_hit_the_cache_not_the_api():
    pytest.importorskip("PIL")
    c = FakeAnthropic()
    v = vlm(c)
    a, b = v.classify([FRAME], PROC), v.classify([FRAME], PROC)
    assert a == b and len(c.calls) == 1 and v.budget.calls == 1


def test_answer_is_read_from_the_text_block_after_a_thinking_block():
    pytest.importorskip("PIL")

    class R:
        content = [type("T", (), {"type": "thinking", "thinking": "hmm"})(), type("B", (), {"type": "text", "text": GOOD})()]
        usage = type("U", (), {"input_tokens": 10, "output_tokens": 10})

    c = FakeAnthropic()
    c.create = lambda **kw: R
    assert vlm(c).classify([FRAME], PROC).step_id == "access_prep"


def test_disk_cache_survives_a_new_client_instance(tmp_path):
    pytest.importorskip("PIL")
    c1, c2 = FakeAnthropic(), FakeAnthropic()
    a = vlm(c1, cache_dir=tmp_path).classify([FRAME], PROC)
    b = vlm(c2, cache_dir=tmp_path).classify([FRAME], PROC)
    assert a == b and len(c1.calls) == 1 and len(c2.calls) == 0


def test_switches_thinking_setting_when_the_model_rejects_disabled_and_remembers_it():
    pytest.importorskip("PIL")

    class Picky(FakeAnthropic):
        def create(self, **kw):
            if kw["thinking"] == {"type": "disabled"}:
                self.calls.append(kw)
                raise RuntimeError('To turn thinking off on this model, send "thinking": {"type": "between_tools"} instead')
            return super().create(**kw)

    c = Picky()
    v = vlm(c)
    v.classify([FRAME], PROC)
    v.classify([FRAME + 1], PROC)
    assert [k["thinking"]["type"] for k in c.calls] == ["disabled", "between_tools", "between_tools"]
    assert v.budget.calls == 2


def test_parse_reading_ignores_text_around_the_json_object():
    r = parse_reading("Here you go:\n" + GOOD + "\nLet me know if you need more.", PROC)
    assert r.step_id == "access_prep" and r.tool == "vial"
