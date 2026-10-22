from __future__ import annotations
import io
import json
import os
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from awareness.procedure import ProcedureSpec


@dataclass
class VLMReading:
    step_id: str
    confidence: float
    rationale: str


class VLMClient(Protocol):
    def classify(self, frame: np.ndarray, procedure: ProcedureSpec) -> VLMReading: ...


class ScriptedVLM:
    """Returns canned readings in order (repeats the last one). Used in tests and offline demos."""

    def __init__(self, readings: list[VLMReading]):
        self._readings = readings
        self._i = 0

    def classify(self, frame: np.ndarray, procedure: ProcedureSpec) -> VLMReading:
        r = self._readings[min(self._i, len(self._readings) - 1)]
        self._i += 1
        return r


class GeminiVLM:
    """Gemini vision client. Needs `google-genai`, `pillow` and GOOGLE_API_KEY; imported lazily so tests need none of them."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model

    def classify(self, frame: np.ndarray, procedure: ProcedureSpec) -> VLMReading:
        from google import genai
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG")
        steps = "\n".join(f"- {s.step_id}: {s.description}" for s in procedure.steps)
        prompt = (
            "You watch a cardiac cath lab. Which procedure step is the clinician performing in this frame?\n"
            f"{steps}\nAnswer JSON only: "
            '{"step_id": <one of the ids>, "confidence": <0-1>, "rationale": <one sentence>}'
        )
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(
            model=self.model,
            contents=[prompt, genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")],
        )
        text = resp.text.strip().removeprefix("```json").removesuffix("```")
        data = json.loads(text)
        return VLMReading(data["step_id"], float(data["confidence"]), data["rationale"])
