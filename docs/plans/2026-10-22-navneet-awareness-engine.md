# Awareness Engine (Proactive Assist) — Implementation Plan

> Owner: Navneet · Branch: `feat/navneet-awareness-engine` · Spec: §5.1 of the [design spec](../specs/2026-10-22-cathlab-learning-loop-design.md) · Task file: [`docs/tasks/navneet.md`](../tasks/navneet.md)

**Goal:** Deliver N1 (procedure model), N3 (`AwarenessEngine`) and N4 (propose → confirm → emit loop). The robot side stays a black box: the only output is an `instruction: str`.

**Scope of this plan:** N1, N3, N4 with the VLM mocked. Voice (N2) and the sim demo (N5) get their own plans. The assistant takes `confirm` and `speak` as plain callables, so N2 plugs in later without changing anything here.

**Tech:** Python ≥3.10, pydantic, pytest. The Gemini VLM client is a lazy import (`google-genai`). Tests never touch network, GPU or Isaac.

## Design decisions

| Decision | Why |
|---|---|
| Procedure steps are a YAML file validated by a pydantic `ProcedureSpec` living in `awareness/`, not `contracts/` | Only awareness consumes it, and contracts are frozen |
| Every tool name in the YAML must exist in `TaskSpec.objects` (checked by `ProcedureSpec.validate_against(task)`) | The proposed instruction must always be something the policy was trained on |
| `next_tool` = first tool of the *following* step | At "sheath insertion" the robot offers the guidewire (acceptance criterion) |
| The VLM only classifies the current step (`VLMReading`); the next tool comes from the state machine | Cheap, deterministic, and testable. The VLM never invents a tool |
| The engine only lets the step stay or advance by one, and ignores low-confidence readings | One noisy frame must not flip the plan |
| `Assistant` takes `confirm: Callable[[str], bool]` and `speak: Callable[[str], None]` | N2 voice yes/no and TTS plug in later; tests pass lambdas |
| Cooldown + confidence threshold + per-step decline memory | Avoid nagging (acceptance in task file) |

## File structure

| File | Responsibility |
|---|---|
| `configs/procedures/coronary_angiography.yaml` | Ordered steps and the tools each needs (N1) |
| `awareness/__init__.py` | Public exports |
| `awareness/procedure.py` | `ProcedureStep`, `ProcedureSpec`, `next_tool()`, `validate_against()` |
| `awareness/vlm.py` | `VLMReading`, `VLMClient` protocol, `ScriptedVLM` (tests), `GeminiVLM` (lazy) |
| `awareness/engine.py` | `AwarenessState`, `AwarenessEngine.observe(frame, t)` (N3) |
| `awareness/assistant.py` | `Assistant.run(stream) -> Iterator[str]` (N4) |
| `awareness/README.md` | Usage and interfaces |
| `tests/test_awareness_procedure.py`, `tests/test_awareness_engine.py`, `tests/test_awareness_assistant.py` | Tests |

Interfaces consumed: `contracts.TaskSpec` (`.objects`, `.instructions(obj)`), `contracts.load_yaml`. Frames are `np.ndarray` (H, W, 3) uint8, as returned by `SimBackend.render()`.

---

## Task 1: Procedure model (N1)

**Files:** create `configs/procedures/coronary_angiography.yaml`, `awareness/__init__.py`, `awareness/procedure.py`; test `tests/test_awareness_procedure.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_awareness_procedure.py
from pathlib import Path
import pytest
from contracts import TaskSpec, load_yaml
from awareness.procedure import ProcedureSpec

ROOT = Path(__file__).resolve().parents[1]
PROC = load_yaml(ROOT / "configs/procedures/coronary_angiography.yaml", ProcedureSpec)
TASK = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)


def test_steps_are_ordered_and_unique():
    ids = [s.step_id for s in PROC.steps]
    assert len(ids) == len(set(ids)) >= 5


def test_all_tools_exist_in_task():
    PROC.validate_against(TASK)


def test_unknown_tool_is_rejected():
    bad = TASK.model_copy(update={"objects": ["gauze"]})
    with pytest.raises(ValueError, match="sheath"):
        PROC.validate_against(bad)


def test_next_tool_is_first_tool_of_following_step():
    assert PROC.next_tool("sheath_insertion") == "guidewire"


def test_next_tool_is_none_at_last_step():
    assert PROC.next_tool(PROC.steps[-1].step_id) is None


def test_unknown_step_raises():
    with pytest.raises(KeyError):
        PROC.next_tool("nope")
```

- [ ] **Step 2: Run, expect failure**

`uv run pytest tests/test_awareness_procedure.py -q` → `ModuleNotFoundError: awareness`.

- [ ] **Step 3: Implement**

```yaml
# configs/procedures/coronary_angiography.yaml
procedure_id: coronary_angiography
steps:
  - {step_id: access_prep, description: "Clinician preps and drapes the radial/femoral access site", tools: [gauze]}
  - {step_id: sheath_insertion, description: "Clinician inserts the introducer sheath into the artery", tools: [sheath]}
  - {step_id: guidewire_advance, description: "Clinician advances the guidewire through the sheath", tools: [guidewire]}
  - {step_id: catheter_advance, description: "Clinician tracks the catheter over the guidewire", tools: [catheter]}
  - {step_id: contrast_injection, description: "Clinician injects contrast with the syringe for imaging", tools: [syringe]}
  - {step_id: closure, description: "Clinician removes devices, cuts sutures and applies pressure", tools: [scissors, gauze]}
```

```python
# awareness/procedure.py
from pydantic import BaseModel
from contracts import TaskSpec


class ProcedureStep(BaseModel):
    step_id: str
    description: str
    tools: list[str]


class ProcedureSpec(BaseModel):
    procedure_id: str
    steps: list[ProcedureStep]

    def index(self, step_id: str) -> int:
        for i, s in enumerate(self.steps):
            if s.step_id == step_id:
                return i
        raise KeyError(step_id)

    def next_tool(self, step_id: str) -> str | None:
        i = self.index(step_id)
        if i + 1 >= len(self.steps):
            return None
        return self.steps[i + 1].tools[0]

    def validate_against(self, task: TaskSpec) -> None:
        unknown = sorted({t for s in self.steps for t in s.tools} - set(task.objects))
        if unknown:
            raise ValueError(f"procedure tools not in task objects: {', '.join(unknown)}")
```

```python
# awareness/__init__.py
from awareness.procedure import ProcedureSpec, ProcedureStep

__all__ = ["ProcedureSpec", "ProcedureStep"]
```

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_awareness_procedure.py -q`
- [ ] **Step 5: Commit** `feat(awareness): add coronary angiography procedure model`

---

## Task 2: VLM interface and step-tracking engine (N3)

**Files:** create `awareness/vlm.py`, `awareness/engine.py`; modify `awareness/__init__.py`; test `tests/test_awareness_engine.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_awareness_engine.py
from pathlib import Path
import numpy as np
from contracts import load_yaml
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import ScriptedVLM, VLMReading

ROOT = Path(__file__).resolve().parents[1]
PROC = load_yaml(ROOT / "configs/procedures/coronary_angiography.yaml", ProcedureSpec)
FRAME = np.zeros((224, 224, 3), dtype=np.uint8)


def engine(*readings, min_confidence=0.5):
    return AwarenessEngine(PROC, ScriptedVLM(list(readings)), min_confidence=min_confidence)


def test_sheath_insertion_predicts_guidewire():
    e = engine(VLMReading("sheath_insertion", 0.9, "sheath entering vessel"))
    s = e.observe(FRAME, 0.0)
    assert (s.step, s.next_tool) == ("sheath_insertion", "guidewire")
    assert s.rationale == "sheath entering vessel"


def test_low_confidence_before_any_step_gives_no_step():
    s = engine(VLMReading("closure", 0.2, "blurry")).observe(FRAME, 0.0)
    assert s.step is None and s.next_tool is None


def test_low_confidence_keeps_previous_step():
    e = engine(VLMReading("guidewire_advance", 0.9, "a"), VLMReading("closure", 0.3, "b"))
    e.observe(FRAME, 0.0)
    s = e.observe(FRAME, 1.0)
    assert s.step == "guidewire_advance"


def test_regression_is_ignored():
    e = engine(VLMReading("catheter_advance", 0.9, "a"), VLMReading("access_prep", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "catheter_advance"


def test_skipping_ahead_is_limited_to_one_step():
    e = engine(VLMReading("sheath_insertion", 0.9, "a"), VLMReading("closure", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "sheath_insertion"


def test_advancing_one_step_is_accepted():
    e = engine(VLMReading("sheath_insertion", 0.9, "a"), VLMReading("guidewire_advance", 0.9, "b"))
    e.observe(FRAME, 0.0)
    assert e.observe(FRAME, 1.0).step == "guidewire_advance"
```

- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError: awareness.engine`).
- [ ] **Step 3: Implement**

```python
# awareness/vlm.py
from __future__ import annotations
import base64
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
    """Gemini vision client. Needs `google-genai` and GOOGLE_API_KEY; imported lazily so tests never need either."""

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
        data = json.loads(resp.text.strip().removeprefix("```json").removesuffix("```"))
        return VLMReading(data["step_id"], float(data["confidence"]), data["rationale"])
```

Drop the unused `base64` import when writing the file.

```python
# awareness/engine.py
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
```

Export `AwarenessEngine`, `AwarenessState`, `VLMReading`, `ScriptedVLM` from `awareness/__init__.py`.

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_awareness_engine.py -q`
- [ ] **Step 5: Commit** `feat(awareness): add VLM step-tracking engine`

---

## Task 3: Proactive assistant (N4)

**Files:** create `awareness/assistant.py`; modify `awareness/__init__.py`; test `tests/test_awareness_assistant.py`.

Behaviour: when the engine's `next_tool` is set and confidence ≥ threshold, propose it (via `speak` and `confirm`). On yes, yield `task.instructions(tool)[0]` and never re-offer that tool. On no, stay quiet for that tool until the step changes. Never propose twice within `cooldown_s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_awareness_assistant.py
from pathlib import Path
import numpy as np
from contracts import TaskSpec, load_yaml
from awareness.assistant import Assistant
from awareness.engine import AwarenessEngine
from awareness.procedure import ProcedureSpec
from awareness.vlm import ScriptedVLM, VLMReading

ROOT = Path(__file__).resolve().parents[1]
PROC = load_yaml(ROOT / "configs/procedures/coronary_angiography.yaml", ProcedureSpec)
TASK = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)
FRAME = np.zeros((224, 224, 3), dtype=np.uint8)


def make(readings, confirm, **kw):
    spoken = []
    engine = AwarenessEngine(PROC, ScriptedVLM(readings))
    a = Assistant(engine, TASK, confirm=confirm, speak=spoken.append, **kw)
    return a, spoken


def stream(n):
    return [(float(i * 10), FRAME) for i in range(n)]


def test_proposes_guidewire_at_sheath_insertion_and_emits_on_yes():
    a, spoken = make([VLMReading("sheath_insertion", 0.9, "r")], confirm=lambda q: True)
    out = list(a.run(stream(1)))
    assert out == ["Hand me the guidewire"]
    assert "guidewire" in spoken[0]


def test_declined_tool_is_not_reoffered_in_same_step():
    asked = []
    a, _ = make([VLMReading("sheath_insertion", 0.9, "r")], confirm=lambda q: asked.append(q) or False)
    assert list(a.run(stream(5))) == []
    assert len(asked) == 1


def test_accepted_tool_is_not_reoffered():
    asked = []
    a, _ = make([VLMReading("sheath_insertion", 0.9, "r")], confirm=lambda q: asked.append(q) or True)
    assert len(list(a.run(stream(5)))) == 1
    assert len(asked) == 1


def test_cooldown_blocks_back_to_back_proposals():
    asked = []
    readings = [VLMReading("sheath_insertion", 0.9, "a"), VLMReading("guidewire_advance", 0.9, "b")]
    a, _ = make(readings, confirm=lambda q: asked.append(q) or False, cooldown_s=30)
    list(a.run([(0.0, FRAME), (5.0, FRAME)]))
    assert len(asked) == 1


def test_no_proposal_when_step_unknown():
    a, _ = make([VLMReading("closure", 0.1, "r")], confirm=lambda q: True)
    assert list(a.run(stream(3))) == []


def test_no_proposal_at_last_step():
    a, _ = make([VLMReading("closure", 0.9, "r")], confirm=lambda q: True)
    assert list(a.run(stream(3))) == []
```

- [ ] **Step 2: Run, expect failure** (`ModuleNotFoundError: awareness.assistant`).
- [ ] **Step 3: Implement**

```python
# awareness/assistant.py
from typing import Callable, Iterable, Iterator
import numpy as np
from contracts import TaskSpec
from awareness.engine import AwarenessEngine


class Assistant:
    def __init__(
        self,
        engine: AwarenessEngine,
        task: TaskSpec,
        confirm: Callable[[str], bool],
        speak: Callable[[str], None] = lambda _: None,
        cooldown_s: float = 20.0,
    ):
        self.engine = engine
        self.task = task
        self.confirm = confirm
        self.speak = speak
        self.cooldown_s = cooldown_s
        self._last_proposal_t: float | None = None
        self._accepted: set[str] = set()
        self._declined: set[tuple[str, str]] = set()

    def run(self, stream: Iterable[tuple[float, np.ndarray]]) -> Iterator[str]:
        for t, frame in stream:
            state = self.engine.observe(frame, t)
            tool = state.next_tool
            if tool is None or tool in self._accepted or (state.step, tool) in self._declined:
                continue
            if self._last_proposal_t is not None and t - self._last_proposal_t < self.cooldown_s:
                continue
            self._last_proposal_t = t
            question = f"Shall I hand you the {tool}?"
            self.speak(question)
            if self.confirm(question):
                self._accepted.add(tool)
                yield self.task.instructions(tool)[0]
            else:
                self._declined.add((state.step, tool))
```

Export `Assistant` from `awareness/__init__.py`.

- [ ] **Step 4: Run, expect pass.** `uv run pytest tests/test_awareness_assistant.py -q`
- [ ] **Step 5: Commit** `feat(awareness): add propose-confirm-emit assistant`

---

## Task 4: README, decisions, full check

- [ ] **Step 1:** Write `awareness/README.md`: purpose, the `instruction: str` interface, how to swap `ScriptedVLM` for `GeminiVLM`, and the `Assistant(engine, task, confirm, speak)` usage example.
- [ ] **Step 2:** Add rows to `docs/decisions.md` (one line each): next tool is derived from the state machine, not the VLM; the engine rejects regressions and skips; the VLM client is pluggable and lazy.
- [ ] **Step 3:** Run `uv run pytest -q` (all green, no network).
- [ ] **Step 4:** Commit `docs(awareness): add README and decisions`, push `feat/navneet-awareness-engine`, and open a PR titled `plan(navneet): awareness engine` for the plan-review sync. Commit authored by you only (no AI trailer, per `AGENTS.md`).

## Open points for Rakesh's plan review

1. Is a `vlm = ["google-genai", "pillow"]` extra in `pyproject.toml` OK? It's a shared file, so I'd raise it in the PR rather than edit it silently.
2. `Assistant.run` yields instructions. Should it send them straight to `serve.client.PolicyClient` (Sharan, not yet merged), or should the demo glue code do that? This plan assumes the glue code does it.
3. Which VLM does the team standardise on? The judge already uses Gemini via `GOOGLE_API_KEY`, so I default to Gemini for the shared key. Latency still needs measuring against the alternatives (Qwen-VL, GPT-4o).
