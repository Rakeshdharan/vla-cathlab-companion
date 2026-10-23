from collections import deque
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
    observed_tool: str | None = None
    corroborated: bool = False


class AwarenessEngine:
    """Tracks the procedure step from a sliding window of frames. The state machine alone picks `next_tool`;
    a tool the VLM sees that belongs to the current step only raises confidence (`corroborated`)."""

    def __init__(
        self,
        procedure: ProcedureSpec,
        vlm: VLMClient,
        min_confidence: float = 0.6,
        window: int = 4,
        min_interval_s: float = 0.0,
        tool_boost: float = 0.1,
        forward_after: int = 2,
    ):
        self.procedure = procedure
        self.vlm = vlm
        self.min_confidence = min_confidence
        self.min_interval_s = min_interval_s
        self.tool_boost = tool_boost
        self.forward_after = forward_after
        self._pending: tuple[str, int] | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=window)
        self._last_query_t: float | None = None
        self._step: str | None = None
        self._confidence = 0.0
        self._rationale = ""
        self._observed_tool: str | None = None
        self._corroborated = False
        self._known = {s.step_id for s in procedure.steps}

    def observe(self, frame: np.ndarray, t: float) -> AwarenessState:
        self._frames.append(frame)
        if self._last_query_t is None or t - self._last_query_t >= self.min_interval_s:
            self._last_query_t = t
            self._update(self.vlm.classify(list(self._frames), self.procedure))
        nxt = self.procedure.next_tool(self._step) if self._step else None
        return AwarenessState(self._step, self._confidence, nxt, self._rationale, t, self._observed_tool, self._corroborated)

    def _update(self, r) -> None:
        if r.step_id not in self._known:
            return
        step = self.procedure.steps[self.procedure.index(r.step_id)]
        corroborated = r.tool is not None and r.tool in step.tools
        confidence = min(1.0, r.confidence + self.tool_boost) if corroborated else r.confidence
        if confidence < self.min_confidence:
            self._pending = None
            return
        if self._step is not None and r.step_id != self._step:
            if self.procedure.index(r.step_id) < self.procedure.index(self._step):
                self._pending = None
                return
            count = self._pending[1] + 1 if self._pending and self._pending[0] == r.step_id else 1
            self._pending = (r.step_id, count)
            if count < self.forward_after:
                return
        self._pending = None
        self._step, self._rationale, self._confidence = r.step_id, r.rationale, confidence
        self._observed_tool, self._corroborated = r.tool, corroborated
