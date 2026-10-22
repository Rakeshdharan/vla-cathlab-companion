# CathLab Learning Loop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the shared foundation of the CathLab learning loop:
- frozen contracts;
- a dummy end-to-end loop (policy → sim → eval report → model card);
- per-member workstream task files, so 5 people and their agents can build stages 01–05 plus the add-ons in parallel.

**Architecture:** A contract-first Python monorepo (spec: [`docs/specs/2026-10-22-cathlab-learning-loop-design.md`](../specs/2026-10-22-cathlab-learning-loop-design.md)).
- Stages exchange versioned artifacts only: LeRobot datasets plus `manifest.yaml`, `EvalReport` JSON, and `ModelCard` YAML.
- Models implement `PolicyAdapter`; simulators implement `SimBackend`.
- `DummySim` and `RandomPolicy` let every workstream integrate from hour 2 without Isaac or GPUs.

**Tech Stack:** Python ≥3.10, `uv`, `pydantic>=2`, `pyyaml`, `numpy`, `pytest`. Each workstream adds its own heavy dependencies (LeRobot, Isaac Sim/Lab, RoboSimStudio, W&B, ElevenLabs) as optional extras.

## Global Constraints

- Robot: Franka FR3 Duo (bimanual); action/obs spec confirmed from provided docs in H0–2 and recorded in `configs/robots/fr3_duo.yaml`.
- Provided datasets: LeRobot v3.0, 16-D action/state, 3 cameras; box handover (94 eps, 30 fps) + cube stacking (490 eps, 6 fps). See `docs/data.md`. Provided environment: Isaac Sim / USD.
- Every dataset carries a `manifest.yaml` with `source` ∈ {`provided`, `domain_rand`, `mimicgen`, `scripted`, `teleop`, `cosmos`}.
- Task stages are exactly: `reach`, `grasp`, `lift`, `transport`, `place`.
- Experiment tracking: Weights & Biases (project `cathlab-companion`).
- Contracts (`contracts/`) change only via PR reviewed by Rakesh.
- Branches: `feat/<owner>-<topic>`; small PRs to `main`; each package has a README and at least one smoke test.
- Promotion rule: v(n+1) is promoted only if its success rate beats v(n) beyond the 95% CI and its p95 latency ≤ the budget in `configs/serve.yaml`.

## File Structure (foundation)

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package definition, optional extras per workstream, pytest config |
| `contracts/__init__.py` | Re-exports public contract types |
| `contracts/specs.py` | `TaskSpec`, `SceneSpec`, `DatasetManifest`, YAML I/O, `STAGES` |
| `contracts/policy.py` | `PolicyAdapter` ABC, `PolicyMeta`, `RandomPolicy` |
| `contracts/sim.py` | `SimBackend` ABC, `StepInfo`, `DummySim` |
| `contracts/reports.py` | `EpisodeResult`, `EvalReport` (success rate, Wilson CI, stage-failure histogram), `ModelCard` |
| `eval/runner.py` | `run_episode`, `run_suite`: drive any policy in any backend → `EvalReport` |
| `configs/tasks/handover.yaml` | Reference "hand me the {object}" task |
| `configs/scenes/dummy.yaml` | Scene for the dummy backend |
| `tests/test_contracts.py`, `tests/test_runner.py` | Foundation tests |
| `docs/tasks/*.md` | Per-member workstream briefs |

---

### Task 1: Project skeleton and spec contracts

**Files:**
- Create: `pyproject.toml`, `contracts/__init__.py`, `contracts/specs.py`, `configs/tasks/handover.yaml`, `configs/scenes/dummy.yaml`
- Test: `tests/test_contracts.py`

**Interfaces:**
- Produces:
  - `STAGES: tuple[str, ...]`;
  - `TaskSpec(task_id, instruction_templates, objects, success_predicate, stage_predicates)` with `.instructions(obj) -> list[str]`;
  - `SceneSpec(scene_id, backend, asset, objects, cameras, randomization)`;
  - `DatasetManifest(dataset_id, version, source, task_id, scene_id, n_episodes, created_by, parent_versions, quality_score, notes)`;
  - `load_yaml(path, model)`, `save_yaml(obj, path)`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "cathlab-companion"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["pydantic>=2", "pyyaml", "numpy"]

[project.optional-dependencies]
train = ["lerobot", "wandb"]
voice = ["elevenlabs"]

[dependency-groups]
dev = ["pytest"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 2: Write the failing test** `tests/test_contracts.py`

```python
from pathlib import Path
import pytest
from pydantic import ValidationError
from contracts import STAGES, TaskSpec, SceneSpec, DatasetManifest, load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[1]

def test_stages_fixed():
    assert STAGES == ("reach", "grasp", "lift", "transport", "place")

def test_task_spec_loads_and_formats():
    task = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)
    assert "guidewire" in task.objects
    assert "hand me the guidewire" in [s.lower() for s in task.instructions("guidewire")]

def test_task_spec_rejects_unknown_stage():
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t", instruction_templates=["x {object}"], objects=["a"],
                 success_predicate="p", stage_predicates={"fly": "p"})

def test_scene_spec_loads():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    assert scene.backend == "dummy"

def test_manifest_roundtrip(tmp_path):
    m = DatasetManifest(dataset_id="cathlab/handover", version="v0", source="provided",
                        task_id="handover", scene_id="cathlab_v1", n_episodes=500, created_by="rakesh")
    p = tmp_path / "manifest.yaml"
    save_yaml(m, p)
    assert load_yaml(p, DatasetManifest) == m

def test_manifest_rejects_unknown_source():
    with pytest.raises(ValidationError):
        DatasetManifest(dataset_id="d", version="v0", source="magic", task_id="t",
                        scene_id="s", n_episodes=1, created_by="x")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'contracts'`

- [ ] **Step 4: Implement `contracts/specs.py`**

```python
from pathlib import Path
from typing import Literal, TypeVar
import yaml
from pydantic import BaseModel, Field

Stage = Literal["reach", "grasp", "lift", "transport", "place"]
STAGES: tuple[str, ...] = ("reach", "grasp", "lift", "transport", "place")
DataSource = Literal["provided", "domain_rand", "mimicgen", "scripted", "teleop", "cosmos"]
Backend = Literal["dummy", "isaac", "robosimstudio"]


class TaskSpec(BaseModel):
    task_id: str
    instruction_templates: list[str]
    objects: list[str]
    success_predicate: str
    stage_predicates: dict[Stage, str]

    def instructions(self, obj: str) -> list[str]:
        return [t.format(object=obj) for t in self.instruction_templates]


class SceneSpec(BaseModel):
    scene_id: str
    backend: Backend
    asset: str
    objects: dict[str, str]
    cameras: list[str]
    randomization: dict[str, list[float]] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    source: DataSource
    task_id: str
    scene_id: str
    n_episodes: int
    created_by: str
    parent_versions: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    notes: str = ""


M = TypeVar("M", bound=BaseModel)


def load_yaml(path: str | Path, model: type[M]) -> M:
    return model.model_validate(yaml.safe_load(Path(path).read_text()))


def save_yaml(obj: BaseModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False))
```

`contracts/__init__.py`:

```python
from contracts.specs import STAGES, Stage, DataSource, TaskSpec, SceneSpec, DatasetManifest, load_yaml, save_yaml

__all__ = ["STAGES", "Stage", "DataSource", "TaskSpec", "SceneSpec", "DatasetManifest", "load_yaml", "save_yaml"]
```

`configs/tasks/handover.yaml`:

```yaml
task_id: handover
instruction_templates:
  - "Hand me the {object}"
  - "Help me with the {object}"
  - "Please pass the {object}"
objects: [guidewire, catheter, sheath, syringe, scissors, gauze]
success_predicate: "target object resting on handover_table and gripper open"
stage_predicates:
  reach: "end-effector within 5 cm of target"
  grasp: "gripper closed and target in contact"
  lift: "target z > table z + 5 cm"
  transport: "target xy within handover_table bounds"
  place: "target resting on handover_table, gripper open"
```

`configs/scenes/dummy.yaml`:

```yaml
scene_id: dummy
backend: dummy
asset: none
objects: {guidewire: none, catheter: none, sheath: none}
cameras: [head, wrist_left, wrist_right]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock contracts configs tests/test_contracts.py
git commit -m "feat(contracts): task/scene/dataset-manifest specs"
```

### Task 2: PolicyAdapter and SimBackend contracts with dummy implementations

**Files:**
- Create: `contracts/policy.py`, `contracts/sim.py`
- Modify: `contracts/__init__.py`
- Test: `tests/test_contracts.py` (append)

**Interfaces:**
- Consumes: `SceneSpec`, `STAGES` from Task 1.
- Produces:
  - `PolicyMeta(name, params_m, chunk_size, control_hz, action_dim)`;
  - `PolicyAdapter.load(ckpt: str) -> None`;
  - `PolicyAdapter.predict(obs: dict[str, np.ndarray], instruction: str) -> np.ndarray` with shape `(chunk_size, action_dim)`;
  - `PolicyAdapter.meta() -> PolicyMeta`;
  - `RandomPolicy(action_dim=16, chunk_size=10, seed=0)`;
  - `StepInfo(stage_flags: dict[str, bool], success: bool, collision: bool = False)`;
  - `SimBackend.reset(scene: SceneSpec, seed: int, target_object: str) -> dict`;
  - `SimBackend.step(action: np.ndarray) -> tuple[dict, StepInfo]`;
  - `SimBackend.render() -> np.ndarray`;
  - `SimBackend.close() -> None`;
  - `DummySim(steps_per_stage=5, fail_at: str | None = None)`.

- [ ] **Step 1: Append the failing tests to `tests/test_contracts.py`**

```python
import numpy as np
from contracts import RandomPolicy, DummySim, StepInfo

def test_random_policy_shape():
    p = RandomPolicy(action_dim=16, chunk_size=10)
    p.load("none")
    a = p.predict({"state": np.zeros(16)}, "hand me the guidewire")
    assert a.shape == (10, 16)
    assert p.meta().action_dim == 16

def test_dummy_sim_progresses_through_stages():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    sim = DummySim(steps_per_stage=2)
    obs = sim.reset(scene, seed=0, target_object="guidewire")
    assert set(obs) >= {"state", "image_head"}
    info = None
    for _ in range(10):
        obs, info = sim.step(np.zeros(16))
    assert isinstance(info, StepInfo) and info.success
    assert all(info.stage_flags[s] for s in STAGES)

def test_dummy_sim_can_fail_at_stage():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    sim = DummySim(steps_per_stage=1, fail_at="grasp")
    sim.reset(scene, seed=0, target_object="sheath")
    for _ in range(20):
        _, info = sim.step(np.zeros(16))
    assert not info.success and info.stage_flags["reach"] and not info.stage_flags["grasp"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'RandomPolicy'`

- [ ] **Step 3: Implement `contracts/policy.py`**

```python
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel


class PolicyMeta(BaseModel):
    name: str
    params_m: float
    chunk_size: int
    control_hz: float
    action_dim: int


class PolicyAdapter(ABC):
    @abstractmethod
    def load(self, ckpt: str) -> None: ...

    @abstractmethod
    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        """Return an action chunk of shape (chunk_size, action_dim)."""

    @abstractmethod
    def meta(self) -> PolicyMeta: ...


class RandomPolicy(PolicyAdapter):
    def __init__(self, action_dim: int = 16, chunk_size: int = 10, seed: int = 0):
        self._meta = PolicyMeta(name="random", params_m=0.0, chunk_size=chunk_size,
                                control_hz=30.0, action_dim=action_dim)
        self._rng = np.random.default_rng(seed)

    def load(self, ckpt: str) -> None:
        pass

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        return self._rng.uniform(-1, 1, (self._meta.chunk_size, self._meta.action_dim))

    def meta(self) -> PolicyMeta:
        return self._meta
```

- [ ] **Step 4: Implement `contracts/sim.py`**

```python
from abc import ABC, abstractmethod
import numpy as np
from pydantic import BaseModel
from contracts.specs import STAGES, SceneSpec


class StepInfo(BaseModel):
    stage_flags: dict[str, bool]
    success: bool
    collision: bool = False


class SimBackend(ABC):
    @abstractmethod
    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]: ...

    @abstractmethod
    def render(self) -> np.ndarray: ...

    def close(self) -> None:
        pass


class DummySim(SimBackend):
    """Deterministic fake: completes one stage every `steps_per_stage` steps, optionally stalls at `fail_at`."""

    def __init__(self, steps_per_stage: int = 5, fail_at: str | None = None):
        self.steps_per_stage = steps_per_stage
        self.fail_at = fail_at
        self._t = 0

    def _obs(self) -> dict[str, np.ndarray]:
        return {"state": np.zeros(16, dtype=np.float32), "image_head": self.render()}

    def reset(self, scene: SceneSpec, seed: int, target_object: str) -> dict[str, np.ndarray]:
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], StepInfo]:
        self._t += 1
        done = min(self._t // self.steps_per_stage, len(STAGES))
        if self.fail_at is not None:
            done = min(done, STAGES.index(self.fail_at))
        flags = {s: i < done for i, s in enumerate(STAGES)}
        return self._obs(), StepInfo(stage_flags=flags, success=all(flags.values()))

    def render(self) -> np.ndarray:
        return np.zeros((224, 224, 3), dtype=np.uint8)
```

Append to `contracts/__init__.py`:

```python
from contracts.policy import PolicyAdapter, PolicyMeta, RandomPolicy
from contracts.sim import SimBackend, StepInfo, DummySim

__all__ += ["PolicyAdapter", "PolicyMeta", "RandomPolicy", "SimBackend", "StepInfo", "DummySim"]
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add contracts tests/test_contracts.py
git commit -m "feat(contracts): PolicyAdapter, SimBackend and dummy implementations"
```

### Task 3: EvalReport and ModelCard contracts

**Files:**
- Create: `contracts/reports.py`
- Modify: `contracts/__init__.py`
- Test: `tests/test_contracts.py` (append)

**Interfaces:**
- Produces:
  - `EpisodeResult(episode, seed, object, instruction, success, failed_stage, steps, latency_ms_p50, latency_ms_p95, scene_variant="default", video_path=None, judge_verdict=None)`;
  - `EvalReport(model_version, dataset_versions, backend, episodes)` with:
    - `.success_rate() -> float`;
    - `.wilson_ci(z=1.96) -> tuple[float, float]`;
    - `.stage_failure_hist() -> dict[str, int]`;
    - `.save(path)`;
    - `EvalReport.load(path)`;
  - `ModelCard(model_name, version, adapter, ckpt_uri, dataset_versions, eval_report_path, wandb_run=None, git_sha=None, promoted=False)`.

- [ ] **Step 1: Append failing tests to `tests/test_contracts.py`**

```python
from contracts import EpisodeResult, EvalReport, ModelCard

def _ep(i, ok, stage=None):
    return EpisodeResult(episode=i, seed=i, object="guidewire", instruction="hand me the guidewire",
                         success=ok, failed_stage=stage, steps=10, latency_ms_p50=20.0, latency_ms_p95=40.0)

def test_eval_report_metrics(tmp_path):
    r = EvalReport(model_version="smolvla@v0", dataset_versions=["cathlab/handover@v0"], backend="dummy",
                   episodes=[_ep(0, True), _ep(1, False, "grasp"), _ep(2, False, "grasp"), _ep(3, True)])
    assert r.success_rate() == 0.5
    lo, hi = r.wilson_ci()
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert r.stage_failure_hist() == {"reach": 0, "grasp": 2, "lift": 0, "transport": 0, "place": 0}
    r.save(tmp_path / "r.json")
    assert EvalReport.load(tmp_path / "r.json") == r

def test_model_card_roundtrip(tmp_path):
    c = ModelCard(model_name="cathlab-pick", version="v0", adapter="smolvla", ckpt_uri="hf://x",
                  dataset_versions=["cathlab/handover@v0"], eval_report_path="reports/v0.json")
    save_yaml(c, tmp_path / "card.yaml")
    assert load_yaml(tmp_path / "card.yaml", ModelCard) == c
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: FAIL with `ImportError: cannot import name 'EpisodeResult'`

- [ ] **Step 3: Implement `contracts/reports.py`**

```python
import json
import math
from pathlib import Path
from pydantic import BaseModel
from contracts.specs import STAGES, Stage


class EpisodeResult(BaseModel):
    episode: int
    seed: int
    object: str
    instruction: str
    success: bool
    failed_stage: Stage | None
    steps: int
    latency_ms_p50: float
    latency_ms_p95: float
    scene_variant: str = "default"
    video_path: str | None = None
    judge_verdict: str | None = None


class EvalReport(BaseModel):
    model_version: str
    dataset_versions: list[str]
    backend: str
    episodes: list[EpisodeResult]

    def success_rate(self) -> float:
        return sum(e.success for e in self.episodes) / max(len(self.episodes), 1)

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        n = len(self.episodes)
        if n == 0:
            return (0.0, 0.0)
        p = self.success_rate()
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def stage_failure_hist(self) -> dict[str, int]:
        hist = {s: 0 for s in STAGES}
        for e in self.episodes:
            if e.failed_stage:
                hist[e.failed_stage] += 1
        return hist

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        return cls.model_validate(json.loads(Path(path).read_text()))


class ModelCard(BaseModel):
    model_name: str
    version: str
    adapter: str
    ckpt_uri: str
    dataset_versions: list[str]
    eval_report_path: str
    wandb_run: str | None = None
    git_sha: str | None = None
    promoted: bool = False
```

Append to `contracts/__init__.py`:

```python
from contracts.reports import EpisodeResult, EvalReport, ModelCard

__all__ += ["EpisodeResult", "EvalReport", "ModelCard"]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add contracts tests/test_contracts.py
git commit -m "feat(contracts): EvalReport with Wilson CI and ModelCard"
```

### Task 4: Evaluation runner (end-to-end smoke loop)

**Files:**
- Create: `eval/__init__.py` (empty), `eval/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: all contracts from Tasks 1–3.
- Produces:
  - `run_episode(policy, sim, scene, task, obj, seed, max_steps=200) -> EpisodeResult`;
  - `run_suite(policy, sim, scene, task, seeds: list[int], model_version: str, dataset_versions: list[str]) -> EvalReport`.

  Akanksh extends these (tiers, video, VLM judge) without changing their signatures.

- [ ] **Step 1: Write failing test** `tests/test_runner.py`

```python
from pathlib import Path
from contracts import DummySim, RandomPolicy, SceneSpec, TaskSpec, load_yaml
from eval.runner import run_episode, run_suite

ROOT = Path(__file__).resolve().parents[1]
SCENE = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
TASK = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)

def test_run_episode_success():
    r = run_episode(RandomPolicy(), DummySim(steps_per_stage=2), SCENE, TASK, "guidewire", seed=0)
    assert r.success and r.failed_stage is None and r.latency_ms_p95 >= r.latency_ms_p50 >= 0

def test_run_episode_reports_failed_stage():
    r = run_episode(RandomPolicy(), DummySim(steps_per_stage=1, fail_at="lift"), SCENE, TASK, "sheath",
                    seed=1, max_steps=30)
    assert not r.success and r.failed_stage == "lift"

def test_run_suite_aggregates():
    rep = run_suite(RandomPolicy(), DummySim(steps_per_stage=1), SCENE, TASK, seeds=[0, 1, 2],
                    model_version="random@v0", dataset_versions=[])
    assert len(rep.episodes) == 3 * len(TASK.objects) and rep.success_rate() == 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.runner'`

- [ ] **Step 3: Implement `eval/runner.py`**

```python
import time
import numpy as np
from contracts import STAGES, EpisodeResult, EvalReport, PolicyAdapter, SceneSpec, SimBackend, TaskSpec


def run_episode(policy: PolicyAdapter, sim: SimBackend, scene: SceneSpec, task: TaskSpec,
                obj: str, seed: int, max_steps: int = 200) -> EpisodeResult:
    instruction = task.instructions(obj)[seed % len(task.instruction_templates)]
    obs = sim.reset(scene, seed=seed, target_object=obj)
    latencies, steps, info = [], 0, None
    while steps < max_steps:
        t0 = time.perf_counter()
        chunk = policy.predict(obs, instruction)
        latencies.append((time.perf_counter() - t0) * 1000)
        for action in chunk:
            obs, info = sim.step(action)
            steps += 1
            if info.success or steps >= max_steps:
                break
        if info.success:
            break
    failed = None if info.success else next(s for s in STAGES if not info.stage_flags[s])
    return EpisodeResult(episode=seed, seed=seed, object=obj, instruction=instruction, success=info.success,
                         failed_stage=failed, steps=steps,
                         latency_ms_p50=float(np.percentile(latencies, 50)),
                         latency_ms_p95=float(np.percentile(latencies, 95)))


def run_suite(policy: PolicyAdapter, sim: SimBackend, scene: SceneSpec, task: TaskSpec, seeds: list[int],
              model_version: str, dataset_versions: list[str]) -> EvalReport:
    episodes = []
    for obj in task.objects:
        for seed in seeds:
            ep = run_episode(policy, sim, scene, task, obj, seed)
            episodes.append(ep.model_copy(update={"episode": len(episodes)}))
    return EvalReport(model_version=model_version, dataset_versions=dataset_versions,
                      backend=scene.backend, episodes=episodes)
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: 14 passed

- [ ] **Step 5: Commit and push foundation**

```bash
git add eval tests/test_runner.py
git commit -m "feat(eval): run_episode/run_suite end-to-end smoke loop"
git push origin main
```

---

## Workstream plans (executed by each owner with their agent)

The detailed workstream tasks live in [`docs/tasks/`](../tasks/README.md). Each owner turns their file into a TDD plan with superpowers:writing-plans in their own branch. Tasks 1–4 above are the integration point for all workstreams and are merged first by Rakesh (target: H2).

## Spec coverage map

| Spec section | Where |
|---|---|
| 3.1 Contracts | Tasks 1–3 |
| 01 Ingest, 03 Fine-tune, Loop controller, Isaac backend, RoboSimStudio sim2sim, demo | [`rakesh.md`](../tasks/rakesh.md) |
| 02 Dataset Intelligence, 05 Version & Serve, literature review / option catalogue | [`sharan.md`](../tasks/sharan.md) |
| 04 Evaluate, VLM judge, model selection, benchmark | Task 4 + [`akanksh.md`](../tasks/akanksh.md) |
| DataEngine, env factory, new objects | [`mitesh.md`](../tasks/mitesh.md) |
| Awareness, voice | [`navneet.md`](../tasks/navneet.md) |
| 7 Timeline, 9 Risks | [`docs/tasks/README.md`](../tasks/README.md) |
