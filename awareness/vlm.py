from __future__ import annotations
import base64
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence
import numpy as np
from awareness.procedure import ProcedureSpec


@dataclass
class VLMReading:
    step_id: str
    confidence: float
    rationale: str
    tool: str | None = None


class VLMClient(Protocol):
    def classify(self, frames: np.ndarray | Sequence[np.ndarray], procedure: ProcedureSpec) -> VLMReading: ...


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class VLMBudget:
    """Hard stop on paid VLM usage; checked before every API call."""

    max_calls: int = 60
    max_tokens: int = 150_000
    calls: int = 0
    tokens: int = 0

    def check(self) -> None:
        if self.calls >= self.max_calls or self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"VLM budget spent: {self.calls}/{self.max_calls} calls, {self.tokens}/{self.max_tokens} tokens")

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.tokens += input_tokens + output_tokens


def as_frames(frames: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    return [frames] if isinstance(frames, np.ndarray) else list(frames)


def build_prompt(procedure: ProcedureSpec, n_frames: int) -> str:
    steps = "\n".join(f"- {s.step_id}: {s.description}" + (f". Visible: {s.cue}" if s.cue else "") for s in procedure.steps)
    tools = ", ".join(sorted({t for s in procedure.steps for t in s.tools}))
    return (
        f"You watch a cardiac cath lab. The {n_frames} image(s) are consecutive frames, oldest first. "
        "Which procedure step is the clinician performing, and which tool are they holding or reaching for?\n"
        f"Steps:\n{steps}\nTools: {tools}\n"
        "Choose access_prep only while the clinician is still preparing or draping and is not using any instrument or device. "
        "If a syringe, bottle, sheath, wire, catheter or gauze is in use, choose the later step it belongs to. "
        "Do not default to the first step: pick the step that matches what the clinician is actively doing.\n"
        'Answer JSON only: {"step_id": <one step id>, "confidence": <0-1>, '
        '"tool": <one tool or null>, "rationale": <at most 15 words>}. No other text.'
    )


def parse_reading(text: str, procedure: ProcedureSpec) -> VLMReading:
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in VLM reply: {text[:80]!r}")
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    step_id = data.get("step_id")
    if step_id not in {s.step_id for s in procedure.steps}:
        raise ValueError(f"unknown step_id from VLM: {step_id!r}")
    tools = {t for s in procedure.steps for t in s.tools}
    tool = data.get("tool")
    return VLMReading(
        step_id,
        min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
        str(data.get("rationale", "")),
        tool if tool in tools else None,
    )


def _jpeg_b64(frame: np.ndarray, max_side: int) -> str:
    from PIL import Image

    img = Image.fromarray(frame)
    scale = max_side / max(img.size)
    if scale < 1:
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


class ScriptedVLM:
    """Returns canned readings in order (repeats the last one). Used in tests and offline demos."""

    def __init__(self, readings: list[VLMReading]):
        self._readings = readings
        self._i = 0

    def classify(self, frames: np.ndarray | Sequence[np.ndarray], procedure: ProcedureSpec) -> VLMReading:
        r = self._readings[min(self._i, len(self._readings) - 1)]
        self._i += 1
        return r


class AnthropicVLM:
    """Claude vision client with cost guardrails: a call/token budget, a frame cap, downscaled JPEGs,
    a small output cap and an in-memory cache. Needs `anthropic`, `pillow` and ANTHROPIC_API_KEY (lazy)."""

    def __init__(
        self,
        model: str = "claude-haiku-5-5",
        client: Any = None,
        budget: VLMBudget | None = None,
        max_frames: int = 4,
        max_side: int = 224,
        max_output_tokens: int = 200,
        cache_dir: str | Path | None = None,
    ):
        self.model = model
        self._client = client
        self.budget = budget or VLMBudget()
        self.max_frames, self.max_side, self.max_output_tokens = max_frames, max_side, max_output_tokens
        self._cache: dict[str, VLMReading] = {}
        self._thinking = "disabled"
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _create(self, content: list[dict]):
        def call():
            return self._client.messages.create(
                model=self.model, max_tokens=self.max_output_tokens, thinking={"type": self._thinking},
                messages=[{"role": "user", "content": content}],
            )

        try:
            return call()
        except Exception as e:  # some models only accept `between_tools` to switch thinking off
            if self._thinking == "disabled" and "between_tools" in str(e):
                self._thinking = "between_tools"
                return call()
            raise

    def classify(self, frames: np.ndarray | Sequence[np.ndarray], procedure: ProcedureSpec) -> VLMReading:
        recent = as_frames(frames)[-self.max_frames :]
        prompt = build_prompt(procedure, len(recent))
        key = hashlib.sha256(b"".join(f.tobytes() for f in recent) + f"{self.model}|{prompt}".encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        if self._cache_dir and (self._cache_dir / f"{key}.json").exists():
            self._cache[key] = VLMReading(**json.loads((self._cache_dir / f"{key}.json").read_text()))
            return self._cache[key]
        self.budget.check()
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        content: list[dict] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(f, self.max_side)}}
            for f in recent
        ]
        content.append({"type": "text", "text": prompt})
        resp = self._create(content)
        self.budget.record(resp.usage.input_tokens, resp.usage.output_tokens)
        text = next((b.text for b in resp.content if hasattr(b, "text")), "")
        reading = parse_reading(text, procedure)
        self._cache[key] = reading
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"{key}.json").write_text(json.dumps(asdict(reading)))
        return reading


class GeminiVLM:
    """Gemini vision client. Needs `google-genai`, `pillow` and GOOGLE_API_KEY; imported lazily so tests need none of them."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    def classify(self, frames: np.ndarray | Sequence[np.ndarray], procedure: ProcedureSpec) -> VLMReading:
        from google import genai

        recent = as_frames(frames)[-4:]
        parts = [genai.types.Part.from_bytes(data=base64.b64decode(_jpeg_b64(f, 224)), mime_type="image/jpeg") for f in recent]
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(model=self.model, contents=[build_prompt(procedure, len(recent)), *parts])
        return parse_reading(resp.text, procedure)
