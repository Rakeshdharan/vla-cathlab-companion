import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from awareness.assistant import Assistant
from awareness.clips import LabeledClip, load_clips
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import VLMClient


@dataclass
class ClipResult:
    clip_id: str
    expected_step: str
    final_step: str | None
    correct: bool
    tool_seen: str | None
    proposals: list[str] = field(default_factory=list)
    latencies_s: list[float] = field(default_factory=list)
    raw_steps: list[str] = field(default_factory=list)


class _Timed:
    def __init__(self, inner: VLMClient):
        self.inner, self.latencies, self.readings = inner, [], []

    def classify(self, frames, procedure):
        t = time.perf_counter()
        try:
            r = self.inner.classify(frames, procedure)
            self.readings.append(r)
            return r
        finally:
            self.latencies.append(time.perf_counter() - t)


def evaluate_clip(clip: LabeledClip, procedure: ProcedureSpec, vlm: VLMClient, fps: float = 1.0) -> ClipResult:
    timed = _Timed(vlm)
    engine = AwarenessEngine(procedure, timed)
    spoken: list[str] = []
    assistant = Assistant(engine, _Task(), confirm=lambda q: True, speak=spoken.append, cooldown_s=0.0)
    list(assistant.run(clip.stream(fps)))
    state = engine._step, engine._observed_tool
    return ClipResult(clip.clip_id, clip.step_id, state[0], state[0] == clip.step_id, state[1], spoken, timed.latencies, [r.step_id for r in timed.readings])


class _Task:
    def instructions(self, obj):
        return [f"Hand me the {obj}"]


def main() -> None:
    from awareness.vlm import AnthropicVLM, VLMBudget

    ap = argparse.ArgumentParser(description="Run labelled clips through the awareness engine with a capped VLM budget")
    ap.add_argument("--clips", default="var/awareness/clips")
    ap.add_argument("--procedure", default="configs/procedures/coronary_angiography.yaml")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=20_000)
    a = ap.parse_args()
    for line in Path(".env").read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip()
    procedure = ProcedureSpec.from_yaml(a.procedure)
    vlm = AnthropicVLM(model=a.model, budget=VLMBudget(max_calls=a.max_calls, max_tokens=a.max_tokens), cache_dir="var/awareness/vlm_cache")
    raw_ok = raw_n = 0
    for clip in load_clips(a.clips):
        r = evaluate_clip(clip, procedure, vlm)
        print(f"{r.clip_id}: expected={r.expected_step} got={r.final_step} correct={r.correct} tool_seen={r.tool_seen}")
        print(f"  asked: {r.proposals}")
        print(f"  raw VLM steps: {r.raw_steps}")
        raw_ok += sum(x == r.expected_step for x in r.raw_steps)
        raw_n += len(r.raw_steps)
        print(f"  latency/call s: {[round(x, 2) for x in r.latencies_s]}")
    print(f"raw per-frame accuracy: {raw_ok}/{raw_n}")
    print(f"budget used: {vlm.budget.calls} calls, {vlm.budget.tokens} tokens")


if __name__ == "__main__":
    main()
