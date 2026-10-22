from dataclasses import dataclass
import numpy as np
from awareness.procedure import ProcedureSpec
from awareness.vlm import VLMClient


@dataclass
class AwarenessState:
    step: str | None
    confidence: float
    next_tool: str | None
    rationale: str
    t: float


class AwarenessEngine:
    def __init__(self, procedure: ProcedureSpec, vlm: VLMClient, min_confidence: float = 0.6):
        self.procedure = procedure
        self.vlm = vlm
        self.min_confidence = min_confidence
        self._step: str | None = None
        self._confidence = 0.0
        self._rationale = ""

    def observe(self, frame: np.ndarray, t: float) -> AwarenessState:
        r = self.vlm.classify(frame, self.procedure)
        if r.confidence >= self.min_confidence and self._accepts(r.step_id):
            self._step, self._confidence, self._rationale = r.step_id, r.confidence, r.rationale
        nxt = self.procedure.next_tool(self._step) if self._step else None
        return AwarenessState(self._step, self._confidence, nxt, self._rationale, t)

    def _accepts(self, step_id: str) -> bool:
        if self._step is None:
            return True
        cur, new = self.procedure.index(self._step), self.procedure.index(step_id)
        return new in (cur, cur + 1)
