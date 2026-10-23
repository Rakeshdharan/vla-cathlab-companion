import hashlib
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from awareness.vlm import BudgetExceeded

TTS_MODEL = "eleven_flash_v2_5"
STT_MODEL = "scribe_v2"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
FORMAT = "mp3_22050_32"


@dataclass
class VoiceBudget:
    """Hard stop on ElevenLabs usage, checked before every API call (cached phrases are free)."""

    max_chars: int = 600
    max_stt_seconds: float = 60.0
    chars: int = 0
    stt_seconds: float = 0.0

    def check_tts(self, text: str) -> None:
        if self.chars + len(text) > self.max_chars:
            raise BudgetExceeded(f"voice budget spent: {self.chars}/{self.max_chars} characters")

    def check_stt(self, seconds: float) -> None:
        if self.stt_seconds + seconds > self.max_stt_seconds:
            raise BudgetExceeded(f"voice budget spent: {self.stt_seconds:.0f}/{self.max_stt_seconds:.0f} s transcribed")


def _client() -> Any:
    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass
    from elevenlabs import ElevenLabs

    return ElevenLabs()


class Speaker:
    """Text to speech with a character budget, a disk cache and an off switch. `speak` returns the mp3 path."""

    def __init__(self, cache_dir: str | Path = "var/awareness/tts", client: Any = None, budget: VoiceBudget | None = None,
                 enabled: bool = True, play: Callable[[Path], None] | None = None, max_phrase_chars: int = 120):
        self.cache_dir, self._client_obj = Path(cache_dir), client
        self.budget, self.enabled, self.play, self.max_phrase_chars = budget or VoiceBudget(), enabled, play, max_phrase_chars

    def speak(self, text: str) -> Path | None:
        if not self.enabled:
            return None
        text = text[: self.max_phrase_chars]
        key = hashlib.sha256(f"{TTS_MODEL}|{VOICE_ID}|{FORMAT}|{text}".encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.mp3"
        if not path.exists():
            self.budget.check_tts(text)
            self._client_obj = self._client_obj or _client()
            audio = self._client_obj.text_to_speech.convert(text=text, voice_id=VOICE_ID, model_id=TTS_MODEL, output_format=FORMAT)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(audio))
            self.budget.chars += len(text)
        if self.play:
            self.play(path)
        return path


class Listener:
    """Speech to text (Scribe) with a seconds budget. Callers pass the clip length so the budget is checked first."""

    def __init__(self, client: Any = None, budget: VoiceBudget | None = None):
        self._client_obj, self.budget = client, budget or VoiceBudget()

    def transcribe(self, audio: bytes, seconds: float) -> str:
        self.budget.check_stt(seconds)
        self._client_obj = self._client_obj or _client()
        result = self._client_obj.speech_to_text.convert(file=io.BytesIO(audio), model_id=STT_MODEL)
        self.budget.stt_seconds += seconds
        return result.text


_YES = re.compile(r"\b(yes|yeah|yep|sure|please|go ahead|do it|correct|ok(ay)?)\b", re.I)
_NO = re.compile(r"\b(no|nope|not now|don't|do not|negative|cancel|stop)\b", re.I)


def parse_yes_no(text: str) -> bool | None:
    """True/False for a clear answer, None when unclear or contradictory (the caller should re-ask or treat as no)."""
    yes, no = bool(_YES.search(text)), bool(_NO.search(text))
    return True if yes and not no else False if no and not yes else None
