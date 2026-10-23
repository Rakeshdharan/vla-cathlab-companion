import io
import pytest
from awareness.vlm import BudgetExceeded
from awareness.voice import Listener, Speaker, VoiceBudget, parse_yes_no


class FakeEL:
    def __init__(self):
        self.tts_calls, self.stt_calls = [], []
        self.text_to_speech = self
        self.speech_to_text = self

    def convert(self, **kw):
        if "text" in kw:
            self.tts_calls.append(kw)
            return iter([b"ab", b"cd"])
        self.stt_calls.append(kw)
        return type("R", (), {"text": "yes please"})()


def test_speak_writes_audio_and_reuses_the_cache(tmp_path):
    c = FakeEL()
    s = Speaker(tmp_path, client=c)
    p1, p2 = s.speak("Doctor, do you need the syringe?"), s.speak("Doctor, do you need the syringe?")
    assert p1 == p2 and p1.read_bytes() == b"abcd" and len(c.tts_calls) == 1
    assert c.tts_calls[0]["model_id"] == "eleven_flash_v2_5" and s.budget.chars == len("Doctor, do you need the syringe?")


def test_character_budget_stops_before_the_api_call(tmp_path):
    c = FakeEL()
    s = Speaker(tmp_path, client=c, budget=VoiceBudget(max_chars=40))
    s.speak("Doctor, do you need the vial?")
    with pytest.raises(BudgetExceeded):
        s.speak("Doctor, do you need the bottle?")
    assert len(c.tts_calls) == 1


def test_disabled_speaker_makes_no_call_and_plays_nothing(tmp_path):
    c, played = FakeEL(), []
    assert Speaker(tmp_path, client=c, enabled=False, play=played.append).speak("hi") is None
    assert not c.tts_calls and not played


def test_play_receives_the_audio_path(tmp_path):
    played = []
    p = Speaker(tmp_path, client=FakeEL(), play=played.append).speak("hi")
    assert played == [p]


def test_listener_checks_seconds_budget_first():
    c = FakeEL()
    lis = Listener(client=c, budget=VoiceBudget(max_stt_seconds=5))
    assert lis.transcribe(b"x", 4) == "yes please"
    with pytest.raises(BudgetExceeded):
        lis.transcribe(b"x", 4)
    assert len(c.stt_calls) == 1


@pytest.mark.parametrize("text,want", [("Yes please", True), ("yeah go ahead", True), ("No, not now", False),
                                       ("yes, no wait", None), ("hmm", None)])
def test_parse_yes_no(text, want):
    assert parse_yes_no(text) is want
