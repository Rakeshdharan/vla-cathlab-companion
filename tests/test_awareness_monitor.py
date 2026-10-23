import numpy as np
from awareness.clips import LabeledClip
from awareness.monitor import run_clip
from awareness.procedure import ProcedureSpec
from awareness.vlm import BudgetExceeded, ScriptedVLM, VLMReading

PROC = ProcedureSpec.from_yaml("configs/procedures/coronary_angiography.yaml")
FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


def clip(n=3):
    return LabeledClip("c0", "access_prep", ["vial"], [FRAME] * n)


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)

        class P:
            name = "abc.mp3"

        return P()


def test_run_clip_reports_each_frame_the_question_and_the_rollout_params():
    sp = FakeSpeaker()
    vlm = ScriptedVLM([VLMReading("access_prep", 0.7, "reaching for vial", tool="vial")])
    out = run_clip(clip(), PROC, vlm, speaker=sp)
    assert [s["frame"] for s in out["steps"]] == [0, 1, 2]
    assert out["steps"][2]["window"] == [0, 1, 2]
    assert out["steps"][0]["reading"]["rationale"] == "reaching for vial"
    assert out["steps"][0]["state"]["next_tool"] == "syringe"
    assert out["questions"] == [{"text": "Doctor, do you need the syringe?", "audio": "abc.mp3", "frame": 0}]
    assert out["handovers"] == [{"object": "syringe", "instruction": "Hand me the syringe"}]
    assert out["correct"] is True and sp.said == ["Doctor, do you need the syringe?"]


def test_run_clip_without_a_speaker_makes_no_audio():
    out = run_clip(clip(1), PROC, ScriptedVLM([VLMReading("access_prep", 0.9, "r")]))
    assert out["questions"][0]["audio"] is None


def test_run_clip_returns_partial_results_when_the_budget_runs_out():
    class Boom:
        calls = 0

        def classify(self, frames, procedure):
            self.calls += 1
            if self.calls > 1:
                raise BudgetExceeded("spent")
            return VLMReading("access_prep", 0.9, "r")

    out = run_clip(clip(3), PROC, Boom())
    assert len(out["steps"]) == 1 and "spent" in out["error"]
