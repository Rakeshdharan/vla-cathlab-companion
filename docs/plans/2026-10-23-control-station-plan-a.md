# Control Station — Plan A: Integration, Orchestrator Core, Agent Harness, UI Shell

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `main` green again, then ship the Control Station foundation:
- the project catalog;
- the control-plane and worker APIs;
- the **agent harness** (task templates → worktree-isolated coding-agent jobs → checks → human review → push);
- the industrial UI shell with working PROJECT, AGENT and LOG tabs.

**Architecture:** Spec: [`docs/specs/2026-10-23-control-station-design.md`](../specs/2026-10-23-control-station-design.md).
- One codebase with two roles. `orchestrator/control.py` (FastAPI) serves the UI, the catalog, runs, jobs and worker health, and mounts the harness router.
- `orchestrator/worker.py` is the same small app started inside each Python environment.
- `harness/` runs coding agents (GitHub Copilot CLI by default) in per-job git worktrees, with a path guard and acceptance checks, and never touches `main` or the user's working folder.
- The frontend is static HTML/CSS/ES-module JS with no build step.

**Tech Stack:** Python ≥3.10, FastAPI ≥0.110, uvicorn, httpx, pydantic 2, PyYAML, SQLite (stdlib), pytest; vanilla JS; git worktrees; Copilot CLI.

**Plan B (separate plan, next):** wiring the scene, generate, intel, train, eval, promote and rollout stages to the workers; policy server and client; eval suite v2; registry; Dockerfiles and compose.

## Global Constraints

- Work in the worktree `/home/ubuntu/wt-cs` on branch `feat/rakesh-control-station`. **Never** edit `/home/ubuntu/vla-cathlab-companion` (another session uses it).
- Commits are authored by the human user only. **No `Co-authored-by` trailers for AI agents.** Run `git config core.hooksPath .githooks` once in the worktree.
- `uv run pytest -q` must stay GPU-, simulator- and network-free. The robosuite tests run separately: `MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m pytest -q tests/sim`.
- Ports: control 8700, robosuite worker 8711, lerobot worker 8712, isaac worker 8713, standalone agent worker 8790.
- Runtime state lives under `var/` (git-ignored). Data and checkpoints never go into git.
- Harness branches are `agent/<task>-<job_id>` from `origin/main`. Agents must not commit or push; the harness commits after the checks with the user's git identity.
- `configs/variation/cathlab_v1.yaml` is frozen (the published `Rakeshdharan56/pick_robosuite@v1` depends on it).
- UI style: grey panels `#e6e8eb`/`#f4f5f7`, navy bars `#2b3a4a`, one accent `#0d6efd`, status lamps, monospace values, no gradients or illustrations.

## File Structure

| Path | Responsibility |
|---|---|
| `contracts/gaps.py` | `GapSpec` (moved out of Mitesh's stub `contracts/__init__.py`) |
| `dataengine/registry.py` | Mitesh's generator registry, with its own `GenerationReceipt` (no clash with `contracts.DatasetManifest` or `dataengine/generate.py`) |
| `configs/catalog.yaml`, `orchestrator/catalog.py` | Catalog of options and status; selection rules |
| `orchestrator/store.py` | SQLite runs and jobs |
| `orchestrator/process.py` | Subprocess with log file, cancel, timeout |
| `orchestrator/worker.py` | Worker app (per environment) |
| `orchestrator/control.py`, `configs/workers.yaml` | Control plane API, worker registry, static UI |
| `harness/model.py`, `harness/guard.py` | Task templates, params, prompt, path guard |
| `harness/gitops.py`, `harness/agents.py`, `configs/agents.yaml`, `harness/runner.py` | Worktrees, agent CLIs, job lifecycle |
| `harness/api.py`, `harness/server.py`, `harness/factory.py`, `harness/__main__.py` | HTTP API, agent worker, default harness, CLI |
| `harness/tasks/*.yaml`, `scripts/assert_expert_rate.py` | 7 starter templates, expert-rate check |
| `frontend/` | `index.html`, `css/station.css`, `js/{api,dom,app,agent}.js` |
| `scripts/demo_control_station.py` | Safe UI demo (throwaway repo, fake agent) |
| `Makefile`, `docs/INSTALL.md`, `docs/HARNESS.md`, `CREDITS.md`, `README.md`, `AGENTS.md` | Dev commands and docs |

---

### Task 0: Integrate the robosuite base into `main` and make the test suite green

**Files:**
- Merge: `origin/feat/rakesh-robosuite-base` (shared sim core, robosuite track, published-dataset docs)
- Modify: `contracts/__init__.py` (take the real one + export `GapSpec`), `contracts/specs.py` (`Backend`), `dataengine/__init__.py`, `dataengine/{domain_rand,mimicgen,scripted_expert,generators}.py`, `tests/test_dataengine.py`, `docs/decisions.md` (union of both sides)
- Create: `contracts/gaps.py`, `dataengine/registry.py`
- Delete: `contracts/types.py` (empty file)

**Interfaces:**
- Produces:
  - `from contracts import GapSpec` (fields `object, region, clutter, stage`), together with all the real contracts exports (`TaskSpec`, `DummySim`, …);
  - `dataengine.registry`: `GenerationReceipt(source, num_episodes)`, `register_method(name)`, `generate(gap, n, method) -> GenerationReceipt`, `list_methods()`;
  - `contracts.specs.Backend = Literal["dummy", "robosuite", "isaac", "mujoco", "robosimstudio"]`.

Why: on `main`, Mitesh's stub replaced `contracts/__init__.py`, so `uv run pytest` fails at collection on 4 files. The shared sim core is still only on the robosuite branch, and Mitesh's `generate()` function in `dataengine/__init__.py` clashes with the shared core's `dataengine/generate.py` module. A dry run of exactly these steps gave **106 passed, 5 skipped** (core) and **8 passed** (robosuite).

- [ ] **Step 1: Start the merge in the worktree**

```bash
cd /home/ubuntu/wt-cs
git config core.hooksPath .githooks
git fetch -q origin
git merge --no-ff --no-commit origin/feat/rakesh-robosuite-base
```

Expected: `CONFLICT (add/add): Merge conflict in contracts/__init__.py` and `CONFLICT (content): Merge conflict in docs/decisions.md`.

- [ ] **Step 2: Take the real contracts and add `GapSpec`**

```bash
git checkout --theirs contracts/__init__.py
git rm -q contracts/types.py
```

Create `contracts/gaps.py`:

```python
"""GapSpec: what the learning loop asks a data generator to produce more of."""
from pydantic import BaseModel


class GapSpec(BaseModel):
    object: str
    region: str
    clutter: str
    stage: str
```

In `contracts/__init__.py`:
- after the line `from contracts.reports import EpisodeResult, EvalReport, ModelCard`, add `from contracts.gaps import GapSpec`;
- in `__all__`, change `"EpisodeResult", "EvalReport", "ModelCard",` to `"EpisodeResult", "EvalReport", "ModelCard", "GapSpec",`.

In `contracts/specs.py` replace the `Backend` line with:

```python
Backend = Literal["dummy", "robosuite", "isaac", "mujoco", "robosimstudio"]
```

- [ ] **Step 3: Move Mitesh's generator registry to `dataengine/registry.py`**

```bash
echo '"""Shared simulator-independent data engine: variation spec, scripted expert, generation, sinks."""' > dataengine/__init__.py
```

Create `dataengine/registry.py`:

```python
"""Registry of named data-generation methods (domain_rand, mimicgen, scripted, teleop, cosmos).

The built-in methods are placeholders that return a GenerationReceipt; the working robosuite generator is
scripts/robosuite_generate.py (see docs/tasks/robosuite-collection.md).
"""
from pydantic import BaseModel
from contracts import GapSpec


class GenerationReceipt(BaseModel):
    source: str
    num_episodes: int


_generators: dict = {}


def register_method(name: str):
    def decorator(func):
        _generators[name] = func
        return func
    return decorator


def generate(gap: GapSpec | None, n: int, method: str) -> GenerationReceipt:
    if method not in _generators:
        raise ValueError(f"Unknown generator method: {method}")
    return _generators[method](gap, n)


def list_methods() -> list[str]:
    return list(_generators.keys())


from dataengine import domain_rand, generators, mimicgen, scripted_expert  # noqa: E402,F401  (register built-ins)
```

Point Mitesh's modules and test at it:

```bash
sed -i -e 's/^from contracts import GapSpec, DatasetManifest, PolicyAdapter$/from contracts import GapSpec, PolicyAdapter\nfrom dataengine.registry import GenerationReceipt/' \
  -e 's/^from contracts import GapSpec, DatasetManifest$/from contracts import GapSpec\nfrom dataengine.registry import GenerationReceipt/' \
  -e 's/^from dataengine import register_method$/from dataengine.registry import register_method/' \
  -e 's/DatasetManifest/GenerationReceipt/g' \
  dataengine/domain_rand.py dataengine/mimicgen.py dataengine/scripted_expert.py dataengine/generators.py
sed -i -e 's/^from contracts import GapSpec, DatasetManifest$/from contracts import GapSpec/' \
  -e 's/^from dataengine import generate, list_methods, register_method$/from dataengine.registry import GenerationReceipt, generate, list_methods, register_method/' \
  -e 's/DatasetManifest/GenerationReceipt/g' tests/test_dataengine.py
```

- [ ] **Step 4: Resolve `docs/decisions.md` by keeping both sides' rows**

```bash
git show :1:docs/decisions.md > /tmp/d_base
git show :2:docs/decisions.md > /tmp/d_ours
git show :3:docs/decisions.md > /tmp/d_theirs
git merge-file --union /tmp/d_ours /tmp/d_base /tmp/d_theirs
cp /tmp/d_ours docs/decisions.md
grep -c '^<<<<<<<\|^>>>>>>>' docs/decisions.md   # expected: 0
```

- [ ] **Step 5: Run both test suites**

Run: `uv run pytest -q`
Expected: `106 passed, 5 skipped` (no collection errors).

Run: `MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m pytest -q tests/sim`
Expected: `8 passed`.

- [ ] **Step 6: Commit the merge and publish `main`**

```bash
git add -A
git commit -m "merge: robosuite base into main; restore contracts exports, GapSpec module, generator registry"
git push origin HEAD:main
```

---

### Task 1: Dependencies, catalog and selection rules

**Files:**
- Modify: `pyproject.toml` (dependencies), `.gitignore` (add `var/`)
- Create: `configs/catalog.yaml`, `orchestrator/__init__.py` (empty), `orchestrator/catalog.py`
- Test: `tests/orchestrator/__init__.py` (empty), `tests/orchestrator/test_catalog.py`

**Interfaces:**
- Produces:
  - `load_catalog(path) -> Catalog` (`Catalog.sections: dict[str, CatalogSection]`, `Catalog.defaults`);
  - `CatalogSection.item(id) -> CatalogItem | None`;
  - `CatalogItem(id, label, status: 'ready'|'in_progress'|'planned', notes, harness_task, requires)`;
  - `selection_problems(catalog, selection: dict) -> list[Problem]`;
  - `Problem(section, item, kind, message, harness_task)`, with `kind` in `unknown_section | missing | unknown_item | not_ready | requires | min | type`.
  - An empty list means the selection is runnable.

- [ ] **Step 1: Add the web dependencies and ignore runtime state**

In `pyproject.toml`, change the dependencies line to:

```toml
dependencies = ["pydantic>=2", "pyyaml", "numpy", "fastapi>=0.110", "uvicorn>=0.29", "httpx>=0.27"]
```

Append to `.gitignore`:

```text

# Control Station runtime state (SQLite, job logs, agent jobs, run snapshots)
var/
```

Run `uv sync` (updates `uv.lock`) and `touch orchestrator/__init__.py tests/orchestrator/__init__.py` after `mkdir -p orchestrator tests/orchestrator`.

- [ ] **Step 2: Write the failing test `tests/orchestrator/test_catalog.py`**

```python
from pathlib import Path
import pytest
from orchestrator.catalog import load_catalog, selection_problems

ROOT = Path(__file__).resolve().parents[2]
CAT = load_catalog(ROOT / "configs/catalog.yaml")


def test_catalog_has_all_sections_and_defaults_are_runnable():
    assert set(CAT.sections) == {"simulator", "embodiment", "objects", "expert", "variation", "model"}
    assert selection_problems(CAT, dict(CAT.defaults)) == []


def test_every_non_ready_item_points_to_a_harness_task():
    for section in CAT.sections.values():
        for item in section.items:
            if item.status != "ready":
                assert item.harness_task, item.id


def test_not_ready_item_reports_harness_task():
    sel = dict(CAT.defaults, simulator="isaac", embodiment="fr3_duo")
    kinds = {(p.item, p.kind): p.harness_task for p in selection_problems(CAT, sel)}
    assert kinds[("isaac", "not_ready")] == "add_sim_backend"
    assert ("fr3_duo", "not_ready") in kinds


def test_requires_rule_blocks_panda_in_isaac():
    sel = dict(CAT.defaults, simulator="isaac")
    assert any(p.item == "panda" and p.kind == "requires" for p in selection_problems(CAT, sel))


def test_unknown_object_points_to_add_object():
    sel = dict(CAT.defaults, objects=["bottle", "guidewire"])
    p = next(p for p in selection_problems(CAT, sel) if p.kind == "unknown_item")
    assert p.item == "guidewire" and p.harness_task == "add_object"


@pytest.mark.parametrize("sel_patch,kind", [
    ({"objects": []}, "min"),
    ({"objects": "bottle"}, "type"),
    ({"extra": "x"}, "unknown_section"),
])
def test_shape_errors(sel_patch, kind):
    sel = dict(CAT.defaults, **sel_patch)
    assert kind in {p.kind for p in selection_problems(CAT, sel)}


def test_missing_section():
    sel = dict(CAT.defaults)
    del sel["model"]
    assert any(p.section == "model" and p.kind == "missing" for p in selection_problems(CAT, sel))
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `uv run pytest -q tests/orchestrator/test_catalog.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.catalog'`

- [ ] **Step 4: Create `configs/catalog.yaml`**

```yaml
catalog_version: 1

# Every option the Control Station offers. status: ready | in_progress | planned.
# Items that are not ready (and `add_task` on a section) point to a harness task template
# in harness/tasks/, so the UI can offer "Build with agent".
sections:
  simulator:
    title: Simulator
    multi: false
    items:
      - {id: robosuite, label: "robosuite (MuJoCo)", status: ready, notes: "Panda, LIBERO I/O, 20 Hz"}
      - {id: isaac, label: "Isaac Sim", status: in_progress, harness_task: add_sim_backend, notes: "Provided FR3 Duo scene, right arm"}
      - {id: mujoco, label: "MuJoCo (standalone)", status: planned, harness_task: add_sim_backend}
  embodiment:
    title: Embodiment
    multi: false
    items:
      - {id: panda, label: "Franka Panda (single arm)", status: ready, requires: {simulator: [robosuite]}}
      - {id: fr3_duo, label: "Franka FR3 Duo (right arm)", status: in_progress, harness_task: add_sim_backend, requires: {simulator: [isaac, mujoco]}}
  objects:
    title: Objects
    multi: true
    min: 1
    add_task: add_object
    items:
      - {id: bottle, label: "Bottle (Ø62 × 250 mm)", status: ready}
      - {id: vial, label: "Vial (Ø27 × 42 mm)", status: ready}
      - {id: syringe, label: "Syringe (140 mm, lying)", status: ready}
  expert:
    title: Expert controller
    multi: false
    items:
      - {id: analytic, label: "Analytic grasp templates + waypoints", status: ready}
      - {id: graspgen, label: "GraspGen grasp ranking", status: planned, harness_task: add_expert}
  variation:
    title: Variation axes
    multi: true
    min: 0
    add_task: add_variation_axis
    items:
      - {id: layout, label: "Object layouts (10)", status: ready}
      - {id: obj_color, label: "Object colour palettes (5)", status: ready}
      - {id: table_color, label: "Table colours (5)", status: ready}
      - {id: mass, label: "Mass × 0.7–1.3", status: ready}
      - {id: friction, label: "Friction 0.5–1.0", status: ready}
      - {id: light, label: "Light intensity / colour temperature", status: ready}
      - {id: camera_jitter, label: "Camera jitter", status: planned, harness_task: add_variation_axis}
      - {id: distractors, label: "Unseen distractor objects", status: planned, harness_task: add_variation_axis}
  model:
    title: Policy model
    multi: false
    items:
      - {id: smolvla, label: "SmolVLA (baseline)", status: ready}
      - {id: act, label: "ACT", status: ready}
      - {id: pi05, label: "π0.5", status: planned, harness_task: add_policy_adapter}

defaults:
  simulator: robosuite
  embodiment: panda
  objects: [bottle, vial, syringe]
  expert: analytic
  variation: [layout, obj_color, table_color, mass, friction, light]
  model: smolvla
```

- [ ] **Step 5: Create `orchestrator/catalog.py`**

```python
"""Project catalog: every option the Control Station offers, with a readiness status."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

Status = Literal["ready", "in_progress", "planned"]
ProblemKind = Literal["unknown_section", "missing", "unknown_item", "not_ready", "requires", "min", "type"]


class CatalogItem(BaseModel):
    id: str
    label: str
    status: Status
    notes: str = ""
    harness_task: str | None = None
    requires: dict[str, list[str]] = Field(default_factory=dict)


class CatalogSection(BaseModel):
    title: str
    multi: bool = False
    min: int = 1
    add_task: str | None = None
    items: list[CatalogItem]

    def item(self, item_id: str) -> CatalogItem | None:
        return next((i for i in self.items if i.id == item_id), None)


class Catalog(BaseModel):
    catalog_version: int
    sections: dict[str, CatalogSection]
    defaults: dict[str, str | list[str]]


class Problem(BaseModel):
    section: str
    item: str | None = None
    kind: ProblemKind
    message: str
    harness_task: str | None = None


def load_catalog(path: str | Path) -> Catalog:
    return Catalog.model_validate(yaml.safe_load(Path(path).read_text()))


def selection_problems(catalog: Catalog, selection: dict) -> list[Problem]:
    """Everything that stops this selection from running. Empty list = runnable."""
    problems: list[Problem] = []
    for key in selection:
        if key not in catalog.sections:
            problems.append(Problem(section=key, kind="unknown_section", message=f"unknown section '{key}'"))
    for name, section in catalog.sections.items():
        if name not in selection:
            problems.append(Problem(section=name, kind="missing", message=f"{section.title}: nothing selected"))
            continue
        value = selection[name]
        if section.multi != isinstance(value, list):
            kind = "a list" if section.multi else "a single id"
            problems.append(Problem(section=name, kind="type", message=f"{section.title}: expected {kind}"))
            continue
        ids = value if section.multi else [value]
        if section.multi and len(ids) < section.min:
            problems.append(Problem(section=name, kind="min", harness_task=section.add_task,
                                    message=f"{section.title}: select at least {section.min}"))
        for item_id in ids:
            item = section.item(item_id)
            if item is None:
                problems.append(Problem(section=name, item=item_id, kind="unknown_item", harness_task=section.add_task,
                                        message=f"{section.title}: unknown item '{item_id}'"))
                continue
            if item.status != "ready":
                problems.append(Problem(section=name, item=item_id, kind="not_ready", harness_task=item.harness_task,
                                        message=f"{item.label} is {item.status.replace('_', ' ')}"))
            for other, allowed in item.requires.items():
                chosen = selection.get(other)
                chosen_ids = chosen if isinstance(chosen, list) else [chosen]
                if not any(c in allowed for c in chosen_ids):
                    problems.append(Problem(section=name, item=item_id, kind="requires",
                                            message=f"{item.label} requires {other} in {allowed}"))
    return problems
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator/test_catalog.py`
Expected: 9 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore configs/catalog.yaml orchestrator tests/orchestrator
git commit -m "feat(orchestrator): project catalog with readiness status and selection rules"
```

### Task 2: SQLite store for runs and jobs

**Files:**
- Create: `orchestrator/store.py`
- Test: `tests/orchestrator/test_store.py`

**Interfaces:**
- Produces:
  - `Store(path)`;
  - `create_run(selection: dict, note='') -> Run`, `get_run(id) -> Run | None`, `list_runs(limit=50) -> list[Run]`;
  - `create_job(kind, name, params=None, run_id=None, worker=None, state='queued') -> Job`;
  - `update_job(id, state=None, result=None, worker=None) -> Job` (merges `result`; raises `KeyError` for an unknown id);
  - `get_job(id) -> Job | None`, `list_jobs(kind=None, run_id=None, limit=100) -> list[Job]` (newest first);
  - models `Run(id, created, selection, note)` and `Job(id, kind, name, run_id, worker, state, params, result, created, updated)`;
  - `new_id(prefix) -> str`.

- [ ] **Step 1: Write the failing test `tests/orchestrator/test_store.py`**

```python
import pytest
from orchestrator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "var" / "control.db")


def test_runs_roundtrip(store):
    run = store.create_run({"simulator": "robosuite", "objects": ["vial"]}, note="first")
    assert run.id.startswith("run-")
    assert store.get_run(run.id) == run
    assert [r.id for r in store.list_runs()] == [run.id]
    assert store.get_run("nope") is None


def test_job_lifecycle_merges_results(store):
    job = store.create_job("stage", "generate", params={"n": 3}, run_id="run-1")
    assert job.state == "queued" and job.params == {"n": 3}
    store.update_job(job.id, state="running", worker="robosuite", result={"episodes": 1})
    done = store.update_job(job.id, state="done", result={"success": 1.0})
    assert done.state == "done" and done.worker == "robosuite"
    assert done.result == {"episodes": 1, "success": 1.0}
    assert done.updated >= done.created


def test_list_jobs_filters_newest_first(store):
    a = store.create_job("stage", "generate", run_id="r1")
    b = store.create_job("agent", "add_object")
    c = store.create_job("stage", "eval", run_id="r2")
    assert [j.id for j in store.list_jobs()] == [c.id, b.id, a.id]
    assert [j.id for j in store.list_jobs(kind="stage")] == [c.id, a.id]
    assert [j.id for j in store.list_jobs(run_id="r1")] == [a.id]


def test_update_unknown_job_raises(store):
    with pytest.raises(KeyError):
        store.update_job("missing", state="done")
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/orchestrator/test_store.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.store'`

- [ ] **Step 3: Create `orchestrator/store.py`**

```python
"""SQLite state for the control plane: runs (frozen selections) and jobs (stage executions)."""
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from pydantic import BaseModel, Field

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL, selection TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, kind TEXT, name TEXT, run_id TEXT, worker TEXT, state TEXT,
    params TEXT, result TEXT, created REAL, updated REAL
);
"""


class Run(BaseModel):
    id: str
    created: float
    selection: dict
    note: str = ""


class Job(BaseModel):
    id: str
    kind: str
    name: str
    run_id: str | None = None
    worker: str | None = None
    state: str
    params: dict = Field(default_factory=dict)
    result: dict = Field(default_factory=dict)
    created: float
    updated: float


def new_id(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # runs ---------------------------------------------------------------
    def create_run(self, selection: dict, note: str = "") -> Run:
        run = Run(id=new_id("run"), created=time.time(), selection=selection, note=note)
        with self._lock, self._db:
            self._db.execute("INSERT INTO runs VALUES (?,?,?,?)",
                             (run.id, run.created, json.dumps(selection), note))
        return run

    def get_run(self, run_id: str) -> Run | None:
        row = self._db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return None if row is None else Run(id=row["id"], created=row["created"],
                                            selection=json.loads(row["selection"]), note=row["note"])

    def list_runs(self, limit: int = 50) -> list[Run]:
        rows = self._db.execute("SELECT id FROM runs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_run(r["id"]) for r in rows]

    # jobs ---------------------------------------------------------------
    def create_job(self, kind: str, name: str, params: dict | None = None, run_id: str | None = None,
                   worker: str | None = None, state: str = "queued") -> Job:
        now = time.time()
        job = Job(id=new_id(kind[:3]), kind=kind, name=name, run_id=run_id, worker=worker, state=state,
                  params=params or {}, result={}, created=now, updated=now)
        with self._lock, self._db:
            self._db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (job.id, kind, name, run_id, worker, state, json.dumps(job.params), "{}", now, now))
        return job

    def update_job(self, job_id: str, state: str | None = None, result: dict | None = None,
                   worker: str | None = None) -> Job:
        job = self.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        merged = {**job.result, **(result or {})}
        with self._lock, self._db:
            self._db.execute("UPDATE jobs SET state=?, result=?, worker=?, updated=? WHERE id=?",
                             (state or job.state, json.dumps(merged), worker or job.worker, time.time(), job_id))
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Job | None:
        row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(id=row["id"], kind=row["kind"], name=row["name"], run_id=row["run_id"], worker=row["worker"],
                   state=row["state"], params=json.loads(row["params"]), result=json.loads(row["result"]),
                   created=row["created"], updated=row["updated"])

    def list_jobs(self, kind: str | None = None, run_id: str | None = None, limit: int = 100) -> list[Job]:
        sql, args = "SELECT id FROM jobs WHERE 1=1", []
        if kind:
            sql, args = sql + " AND kind=?", [*args, kind]
        if run_id:
            sql, args = sql + " AND run_id=?", [*args, run_id]
        rows = self._db.execute(sql + " ORDER BY created DESC LIMIT ?", (*args, limit)).fetchall()
        return [self.get_job(r["id"]) for r in rows]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator/test_store.py`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add orchestrator/store.py tests/orchestrator/test_store.py
git commit -m "feat(orchestrator): SQLite store for runs and jobs"
```

### Task 3: Process runner with log file, cancel and timeout

**Files:**
- Create: `orchestrator/process.py`
- Test: `tests/orchestrator/test_process.py`

**Interfaces:**
- Produces:
  - `start_process(cmd: list[str] | str, log_path, cwd=None, env=None, timeout_s=None, on_exit=None, append=False) -> ProcessHandle` (a string runs through the shell; stdout and stderr go to `log_path`);
  - `ProcessHandle.state` in `running | done | failed | cancelled | timeout`, `.returncode`, `.wait(timeout=None) -> state`, `.cancel()`, `.pid`, `.log_path`;
  - `read_log(path, offset=0, max_bytes=256000) -> (text, next_offset)`.

- [ ] **Step 1: Write the failing test `tests/orchestrator/test_process.py`**

```python
import sys
from orchestrator.process import read_log, start_process

PY = sys.executable


def test_success_writes_log_and_calls_back(tmp_path):
    seen = []
    h = start_process([PY, "-c", "print('hello'); import sys; print('err', file=sys.stderr)"],
                      tmp_path / "a.log", on_exit=lambda rc, st: seen.append((rc, st)))
    assert h.wait(10) == "done" and h.returncode == 0
    text, off = read_log(tmp_path / "a.log")
    assert "hello" in text and "err" in text and off == len(text.encode())
    assert seen == [(0, "done")]


def test_failure_state(tmp_path):
    h = start_process([PY, "-c", "raise SystemExit(3)"], tmp_path / "b.log")
    assert h.wait(10) == "failed" and h.returncode == 3


def test_shell_string_cwd_and_env(tmp_path):
    h = start_process("echo $GREETING > out.txt", tmp_path / "c.log", cwd=tmp_path, env={"GREETING": "hi"})
    assert h.wait(10) == "done"
    assert (tmp_path / "out.txt").read_text().strip() == "hi"


def test_cancel_and_timeout(tmp_path):
    h = start_process([PY, "-c", "import time; time.sleep(30)"], tmp_path / "d.log")
    h.cancel()
    assert h.wait(10) == "cancelled"
    t = start_process([PY, "-c", "import time; time.sleep(30)"], tmp_path / "e.log", timeout_s=0.3)
    assert t.wait(10) == "timeout"


def test_read_log_offsets_and_append(tmp_path):
    p = tmp_path / "f.log"
    start_process([PY, "-c", "print('one')"], p).wait(10)
    first, off = read_log(p)
    start_process([PY, "-c", "print('two')"], p, append=True).wait(10)
    second, off2 = read_log(p, off)
    assert first.strip() == "one" and second.strip() == "two" and off2 > off
    assert read_log(tmp_path / "missing.log", 5) == ("", 5)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/orchestrator/test_process.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.process'`

- [ ] **Step 3: Create `orchestrator/process.py`**

```python
"""Run one subprocess with its output in a log file; used by workers and the agent harness."""
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable


class ProcessHandle:
    def __init__(self, proc: subprocess.Popen, log_path: Path, timeout_s: float | None,
                 on_exit: Callable[[int, str], None] | None):
        self.proc, self.log_path = proc, log_path
        self.state = "running"  # running -> done | failed | cancelled | timeout
        self.returncode: int | None = None
        self._cancelled = False
        self._timed_out = False
        self._on_exit = on_exit
        self._done = threading.Event()
        self._timer = threading.Timer(timeout_s, self._timeout) if timeout_s else None
        if self._timer:
            self._timer.daemon = True
            self._timer.start()
        threading.Thread(target=self._watch, daemon=True).start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    def _kill(self) -> None:
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def _timeout(self) -> None:
        self._timed_out = True
        self._kill()

    def cancel(self) -> None:
        self._cancelled = True
        self._kill()

    def _watch(self) -> None:
        rc = self.proc.wait()
        if self._timer:
            self._timer.cancel()
        self.returncode = rc
        if self._cancelled:
            self.state = "cancelled"
        elif self._timed_out:
            self.state = "timeout"
        else:
            self.state = "done" if rc == 0 else "failed"
        if self._on_exit:
            self._on_exit(rc, self.state)
        self._done.set()

    def wait(self, timeout: float | None = None) -> str:
        self._done.wait(timeout)
        return self.state


def start_process(cmd: list[str] | str, log_path: str | Path, cwd: str | Path | None = None,
                  env: dict[str, str] | None = None, timeout_s: float | None = None,
                  on_exit: Callable[[int, str], None] | None = None, append: bool = False) -> ProcessHandle:
    """Start `cmd` (argv list, or a shell string) with stdout+stderr appended to `log_path`."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab" if append else "wb")
    proc = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, env={**os.environ, **(env or {})},
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    return ProcessHandle(proc, log_path, timeout_s, on_exit)


def read_log(path: str | Path, offset: int = 0, max_bytes: int = 256_000) -> tuple[str, int]:
    """Return (new text since `offset`, next offset). Missing file -> ('', offset)."""
    path = Path(path)
    if not path.exists():
        return "", offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace"), offset + len(data)
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator/test_process.py`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add orchestrator/process.py tests/orchestrator/test_process.py
git commit -m "feat(orchestrator): process runner with log file, cancel and timeout"
```

### Task 4: Worker app (one per Python environment)

**Files:**
- Create: `orchestrator/worker.py`
- Test: `tests/orchestrator/test_worker.py`

**Interfaces:**
- Consumes: Task 3 `start_process`, `read_log`, `ProcessHandle`.
- Produces:
  - `create_worker_app(name, capabilities: list[str], var_dir) -> FastAPI` with:
    - `GET /health` → `{name, capabilities, python, executable, pid, jobs_running}`;
    - `POST /jobs` `{argv, name, cwd, env, timeout_s}` → job view;
    - `GET /jobs/{id}` → `{id, name, state, returncode, started}`;
    - `GET /jobs/{id}/log?offset=` → `{text, offset, state}`;
    - `POST /jobs/{id}/cancel`.
  - CLI: `python -m orchestrator.worker --name N --capabilities a,b --port P [--host] [--var-dir]`.

- [ ] **Step 1: Write the failing test `tests/orchestrator/test_worker.py`**

```python
import sys
import time
from fastapi.testclient import TestClient
from orchestrator.worker import create_worker_app

PY = sys.executable


def wait_state(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/jobs/{job_id}").json()["state"]
        if st != "running":
            return st
        time.sleep(0.05)
    return "running"


def test_health_reports_capabilities(tmp_path):
    c = TestClient(create_worker_app("robosuite", ["generate", "eval"], tmp_path))
    h = c.get("/health").json()
    assert h["name"] == "robosuite" and h["capabilities"] == ["generate", "eval"] and h["jobs_running"] == 0


def test_job_runs_and_log_is_readable_incrementally(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path))
    job = c.post("/jobs", json={"argv": [PY, "-c", "print('stage-ok')"], "name": "generate"}).json()
    assert job["name"] == "generate"
    assert wait_state(c, job["id"]) == "done"
    log = c.get(f"/jobs/{job['id']}/log").json()
    assert "stage-ok" in log["text"] and log["state"] == "done"
    assert c.get(f"/jobs/{job['id']}/log", params={"offset": log["offset"]}).json()["text"] == ""


def test_cancel_unknown_and_empty(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path))
    job = c.post("/jobs", json={"argv": [PY, "-c", "import time; time.sleep(30)"]}).json()
    assert c.post(f"/jobs/{job['id']}/cancel").status_code == 200
    assert wait_state(c, job["id"]) == "cancelled"
    assert c.get("/jobs/nope").status_code == 404
    assert c.post("/jobs", json={"argv": []}).status_code == 400
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/orchestrator/test_worker.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.worker'`

- [ ] **Step 3: Create `orchestrator/worker.py`**

```python
"""Worker app: runs stage commands inside one Python environment (robosuite, lerobot, isaac, ...).

Start: python -m orchestrator.worker --name robosuite --capabilities generate,eval --port 8711
"""
import argparse
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from orchestrator.process import ProcessHandle, read_log, start_process


class JobRequest(BaseModel):
    argv: list[str]
    name: str = "job"
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_s: float | None = None


def create_worker_app(name: str, capabilities: list[str], var_dir: str | Path) -> FastAPI:
    var_dir = Path(var_dir)
    app = FastAPI(title=f"cathlab worker: {name}")
    jobs: dict[str, dict] = {}

    def view(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job {job_id}")
        h: ProcessHandle = job["handle"]
        return {"id": job_id, "name": job["name"], "state": h.state, "returncode": h.returncode,
                "started": job["started"]}

    @app.get("/health")
    def health() -> dict:
        running = sum(1 for j in jobs.values() if j["handle"].state == "running")
        return {"name": name, "capabilities": capabilities, "python": platform.python_version(),
                "executable": sys.executable, "pid": os.getpid(), "jobs_running": running}

    @app.post("/jobs")
    def submit(req: JobRequest) -> dict:
        if not req.argv:
            raise HTTPException(400, "argv must not be empty")
        job_id = uuid.uuid4().hex[:12]
        handle = start_process(req.argv, var_dir / "jobs" / f"{job_id}.log", cwd=req.cwd, env=req.env,
                               timeout_s=req.timeout_s)
        jobs[job_id] = {"handle": handle, "name": req.name, "started": time.time()}
        return view(job_id)

    @app.get("/jobs/{job_id}")
    def get(job_id: str) -> dict:
        return view(job_id)

    @app.get("/jobs/{job_id}/log")
    def log(job_id: str, offset: int = 0) -> dict:
        state = view(job_id)["state"]
        text, new_offset = read_log(jobs[job_id]["handle"].log_path, offset)
        return {"text": text, "offset": new_offset, "state": state}

    @app.post("/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        view(job_id)
        jobs[job_id]["handle"].cancel()
        return view(job_id)

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--capabilities", default="", help="comma-separated, e.g. generate,eval")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--var-dir", default="var/workers")
    args = ap.parse_args()
    caps = [c for c in args.capabilities.split(",") if c]
    app = create_worker_app(args.name, caps, Path(args.var_dir) / args.name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator/test_worker.py`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add orchestrator/worker.py tests/orchestrator/test_worker.py
git commit -m "feat(orchestrator): worker app running stage commands per environment"
```

### Task 5: Harness task model, prompt rendering and path guard

**Files:**
- Create: `harness/__init__.py` (empty), `harness/model.py`, `harness/guard.py`
- Test: `tests/harness/__init__.py` (empty), `tests/harness/test_model.py`

**Interfaces:**
- Produces:
  - `Param(name, type: str|int|float|bool|choice|text, default, choices, required=True, help)`;
  - `Check(name, cmd, timeout_s=900)`;
  - `TaskTemplate(id, title, description, catalog_target, params, context, prompt, allowed_paths, acceptance, timeout_min=30)`;
  - `ParamError`;
  - `load_templates(dir) -> dict[id, TaskTemplate]` (the file name must equal the id);
  - `validate_params(t, params) -> dict` (coerces types and fills defaults);
  - `allowed_for(t, params) -> list[str]` and `checks_for(t, params) -> list[Check]` (fill `{param}` placeholders; comma lists expand);
  - `render_prompt(t, params, extra='') -> str`;
  - `RULES`;
  - `path_violations(files, allowed) -> list[str]` and `is_allowed(path, allowed)` (pattern `dir/` = prefix, otherwise fnmatch).

- [ ] **Step 1: Create the packages**

`mkdir -p harness/tasks tests/harness && touch harness/__init__.py tests/harness/__init__.py`

- [ ] **Step 2: Write the failing test `tests/harness/test_model.py`**

```python
import pytest
from harness.guard import path_violations
from harness.model import Check, Param, ParamError, TaskTemplate, allowed_for, checks_for, render_prompt, validate_params

T = TaskTemplate(
    id="t", title="Add object", description="d",
    params=[Param(name="name"), Param(name="mass", type="float", default=0.1),
            Param(name="grasp", type="choice", choices=["top", "side"], default="top"),
            Param(name="note", type="text", required=False), Param(name="push", type="bool", default=False)],
    context=["AGENTS.md"], prompt="Add {name} ({mass} kg, {grasp} grasp).",
    allowed_paths=["configs/", "sim/robosuite/*.py"], acceptance=[Check(name="t", cmd="pytest -q")])


def test_validate_params_coerces_and_defaults():
    out = validate_params(T, {"name": "hoop", "mass": "0.05", "push": "yes"})
    assert out == {"name": "hoop", "mass": 0.05, "grasp": "top", "note": "", "push": True}


@pytest.mark.parametrize("params,msg", [
    ({}, "missing required"),
    ({"name": "x", "grasp": "pinch"}, "not in"),
    ({"name": "x", "bogus": 1}, "unknown"),
    ({"name": "x", "mass": "heavy"}, "mass"),
])
def test_validate_params_errors(params, msg):
    with pytest.raises(ParamError, match=msg):
        validate_params(T, params)


def test_render_prompt_contains_rules_params_paths_checks_and_extra():
    p = render_prompt(T, validate_params(T, {"name": "hoop"}), extra="Use blue.")
    assert "Add hoop (0.1 kg, top grasp)." in p
    assert "Do NOT run `git commit`" in p and "- AGENTS.md" in p
    assert "- sim/robosuite/*.py" in p and "`pytest -q`" in p and "Use blue." in p


def test_path_guard():
    allowed = ["configs/", "sim/robosuite/*.py", "tests/sim/test_*.py"]
    files = ["configs/a.yaml", "sim/robosuite/env.py", "tests/sim/test_x.py", "contracts/specs.py", "configsX/a"]
    assert path_violations(files, allowed) == ["contracts/specs.py", "configsX/a"]


def test_placeholders_in_allowed_paths_and_checks():
    t = TaskTemplate(id="f", title="Free", description="d", params=[Param(name="paths"), Param(name="obj")],
                     prompt="x", allowed_paths=["{paths}", "docs/decisions.md"],
                     acceptance=[Check(name="c", cmd="check --objects {obj}")])
    params = {"paths": "sim/robosuite/, tests/sim/", "obj": "hoop"}
    assert allowed_for(t, params) == ["sim/robosuite/", "tests/sim/", "docs/decisions.md"]
    assert checks_for(t, params)[0].cmd == "check --objects hoop"
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `uv run pytest -q tests/harness/test_model.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness'`

- [ ] **Step 4: Create `harness/model.py`**

```python
"""Harness task templates: typed parameters, context files, prompt, allowed paths, acceptance checks."""
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field


class Param(BaseModel):
    name: str
    type: Literal["str", "int", "float", "bool", "choice", "text"] = "str"
    default: Any = None
    choices: list[str] = Field(default_factory=list)
    required: bool = True
    help: str = ""


class Check(BaseModel):
    name: str
    cmd: str
    timeout_s: float = 900


class TaskTemplate(BaseModel):
    id: str
    title: str
    description: str
    catalog_target: str = ""
    params: list[Param] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    prompt: str
    allowed_paths: list[str]
    acceptance: list[Check] = Field(default_factory=list)
    timeout_min: float = 30


class ParamError(ValueError):
    pass


def load_templates(directory: str | Path) -> dict[str, TaskTemplate]:
    out = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        t = TaskTemplate.model_validate(yaml.safe_load(path.read_text()))
        if t.id != path.stem:
            raise ValueError(f"{path.name}: id '{t.id}' must equal the file name")
        out[t.id] = t
    return out


def _coerce(p: Param, raw: Any) -> Any:
    if p.type in ("str", "text"):
        return str(raw)
    if p.type == "int":
        return int(raw)
    if p.type == "float":
        return float(raw)
    if p.type == "bool":
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
    if str(raw) not in p.choices:
        raise ParamError(f"{p.name}: '{raw}' not in {p.choices}")
    return str(raw)


def validate_params(template: TaskTemplate, params: dict) -> dict:
    """Coerce to declared types, apply defaults, reject unknown/missing/invalid values."""
    known = {p.name: p for p in template.params}
    unknown = set(params) - set(known)
    if unknown:
        raise ParamError(f"unknown parameter(s): {sorted(unknown)}")
    out = {}
    for p in template.params:
        raw = params.get(p.name, p.default)
        if raw is None or raw == "":
            if p.required:
                raise ParamError(f"missing required parameter '{p.name}'")
            out[p.name] = ""
            continue
        try:
            out[p.name] = _coerce(p, raw)
        except (TypeError, ValueError) as e:
            raise ParamError(f"{p.name}: {e}") from e
    return out


def allowed_for(template: TaskTemplate, params: dict) -> list[str]:
    """Allowed path patterns with {param} placeholders filled; comma-separated values expand to several."""
    out = []
    for pattern in template.allowed_paths:
        out += [x.strip() for x in pattern.format(**params).split(",") if x.strip()]
    return out


def checks_for(template: TaskTemplate, params: dict) -> list[Check]:
    return [Check(name=c.name, cmd=c.cmd.format(**params), timeout_s=c.timeout_s) for c in template.acceptance]


RULES = """You are working in a git worktree of the CathLab Companion repo, on a fresh branch.
Rules:
- Read AGENTS.md and the context files listed below before changing anything.
- Only create or edit files matching the allowed paths below. Any other change fails the job.
- Do NOT run `git commit` or `git push`; the harness commits and a human reviews the diff.
- Never print, commit or hard-code secrets (.env, tokens, keys).
- Keep tests runnable without GPU or network unless the acceptance checks say otherwise.
- Finish by running the acceptance checks yourself and fixing failures."""


def render_prompt(template: TaskTemplate, params: dict, extra: str = "") -> str:
    body = template.prompt.format(**params)
    parts = [
        f"# Task: {template.title}",
        RULES,
        "## Context files (read first)\n" + "\n".join(f"- {c}" for c in template.context),
        "## Parameters\n" + "\n".join(f"- {k}: {v}" for k, v in params.items()) if params else "",
        "## Instructions\n" + body.strip(),
        "## Allowed paths\n" + "\n".join(f"- {a}" for a in allowed_for(template, params)),
        "## Acceptance checks (must pass)\n" + "\n".join(f"- `{c.cmd}`" for c in checks_for(template, params)),
        f"## Additional instructions from the user\n{extra.strip()}" if extra.strip() else "",
    ]
    return "\n\n".join(p for p in parts if p) + "\n"
```

- [ ] **Step 5: Create `harness/guard.py`**

```python
"""Path guard: every changed file must match an allowed pattern.

Pattern forms: 'dir/' = anything under dir; otherwise fnmatch ('*' also matches '/')."""
from fnmatch import fnmatch


def is_allowed(path: str, allowed: list[str]) -> bool:
    for pattern in allowed:
        if pattern.endswith("/") and path.startswith(pattern):
            return True
        if fnmatch(path, pattern):
            return True
    return False


def path_violations(files: list[str], allowed: list[str]) -> list[str]:
    return [f for f in files if not is_allowed(f, allowed)]
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/harness/test_model.py`
Expected: 8 passed

- [ ] **Step 7: Commit**

```bash
git add harness tests/harness
git commit -m "feat(harness): task templates, prompt rendering and path guard"
```

### Task 6: Harness git worktrees, agent CLIs and job runner

**Files:**
- Create: `harness/gitops.py`, `harness/agents.py`, `configs/agents.yaml`, `harness/runner.py`
- Test: `tests/harness/conftest.py`, `tests/harness/test_runner.py`

**Interfaces:**
- Consumes: Task 3 `start_process`, `read_log`; Task 5 model functions and `path_violations`.
- Produces:
  - **gitops:** `git(args, cwd) -> str`, `create_worktree(repo, path, branch, base_ref, fetch=True, remote='origin') -> base_sha`, `stage_all(wt) -> list[str]`, `commit(wt, msg) -> sha`, `diff(wt, base_sha, stat=False) -> str`, `push(wt, branch, remote)`, `remove_worktree(repo, path, branch, delete_branch=True)`, `GitError`.
  - **agents:** `AgentConfig(name, argv, description)` (argv placeholders `{workdir}` `{prompt}` `{prompt_file}`), `AgentsConfig(default, agents).get(name|None)`, `load_agents(path)`, `build_argv(agent, workdir, prompt, prompt_file)`.
  - **runner:** `HarnessSettings(repo, var_dir, worktrees_dir, base_ref='origin/main', remote='origin', fetch=True)` and `Harness(settings, templates, agents)` with:
    - `create(task, params, extra='', agent=None) -> AgentJob`, `run(id) -> AgentJob` (synchronous), `start(id) -> Thread`;
    - `get(id)` (`KeyError` if unknown), `list_jobs()`, `log(id, offset) -> (text, offset)`, `diff(id) -> str`;
    - `push(id)`, `discard(id)`, `job_dir(id)`.
  - `AgentJob(id, task, title, params, extra, agent, branch='agent/<task>-<id>', worktree, state, error, base_sha, head_sha, changed_files, checks: list[CheckResult(name, ok, state)], created, updated)`.
  - States: `created → worktree → agent_running → guard → checks → ready_for_review → pushed | discarded`, or `failed`.
  - `HarnessError`.

- [ ] **Step 1: Write the failing test `tests/harness/conftest.py`**

```python
"""Shared fixtures: a throwaway git repo with an `origin` remote and a fake agent CLI."""
import subprocess
import sys
import textwrap
from pathlib import Path
import pytest
from harness.agents import AgentConfig, AgentsConfig
from harness.model import Check, Param, TaskTemplate
from harness.runner import Harness, HarnessSettings


def sh(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


FAKE_AGENT = textwrap.dedent("""
    import sys, pathlib
    workdir, prompt_file, target = sys.argv[1], sys.argv[2], sys.argv[3]
    prompt = pathlib.Path(prompt_file).read_text()
    print("fake agent read", len(prompt), "chars")
    p = pathlib.Path(workdir) / target
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("made by fake agent\\n")
""")


@pytest.fixture
def repo(tmp_path) -> Path:
    origin = tmp_path / "origin.git"
    sh("git", "init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    work = tmp_path / "repo"
    sh("git", "init", "-q", "-b", "main", str(work), cwd=tmp_path)
    for k, v in (("user.name", "Tester"), ("user.email", "t@example.com")):
        sh("git", "config", k, v, cwd=work)
    (work / "README.md").write_text("hello\n")
    sh("git", "add", "-A", cwd=work)
    sh("git", "commit", "-q", "-m", "init", cwd=work)
    sh("git", "remote", "add", "origin", str(origin), cwd=work)
    sh("git", "push", "-q", "-u", "origin", "main", cwd=work)
    return work


def make_harness(tmp_path: Path, repo: Path, target: str = "configs/new.txt",
                 checks: list[Check] | None = None) -> Harness:
    agent_py = tmp_path / "fake_agent.py"
    agent_py.write_text(FAKE_AGENT)
    agents = AgentsConfig(default="fake", agents={
        "fake": AgentConfig(name="fake", argv=[sys.executable, str(agent_py), "{workdir}", "{prompt_file}", target])})
    template = TaskTemplate(
        id="demo", title="Demo task", description="write a file",
        params=[Param(name="name", type="str"), Param(name="n", type="int", default=2, required=False)],
        context=["README.md"], prompt="Create {name} x{n}.", allowed_paths=["configs/"],
        acceptance=checks if checks is not None else [Check(name="exists", cmd="test -f configs/new.txt")])
    settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
    return Harness(settings, {"demo": template}, agents)
```

- [ ] **Step 2: Write the failing test `tests/harness/test_runner.py`**

```python
import subprocess
import pytest
from harness.model import Check
from harness.runner import HarnessError
from tests.harness.conftest import make_harness


def test_happy_path_ready_for_review_then_push(tmp_path, repo):
    h = make_harness(tmp_path, repo)
    job = h.create("demo", {"name": "hoop"}, extra="be quick")
    assert job.branch == f"agent/demo-{job.id}" and job.params == {"name": "hoop", "n": 2}
    prompt = (h.job_dir(job.id) / "prompt.md").read_text()
    assert "Create hoop x2." in prompt and "be quick" in prompt

    done = h.run(job.id)
    assert done.state == "ready_for_review", done.error
    assert done.changed_files == ["configs/new.txt"]
    assert [c.ok for c in done.checks] == [True]
    assert "made by fake agent" in h.diff(job.id)
    text, _ = h.log(job.id)
    assert "fake agent read" in text and "ready for review" in text
    log = subprocess.run(["git", "log", "-1", "--format=%an|%s", done.branch], cwd=repo,
                         capture_output=True, text=True).stdout.strip()
    assert log == f"Tester|agent(demo): Demo task [{job.id}]"

    pushed = h.push(job.id)
    assert pushed.state == "pushed"
    remote = subprocess.run(["git", "ls-remote", "--heads", "origin", done.branch], cwd=repo,
                            capture_output=True, text=True).stdout
    assert done.branch in remote
    assert [j.id for j in h.list_jobs()] == [job.id]


def test_guard_blocks_changes_outside_allowed_paths(tmp_path, repo):
    h = make_harness(tmp_path, repo, target="contracts/hack.py")
    job = h.run(h.create("demo", {"name": "x"}).id)
    assert job.state == "failed" and "outside allowed paths" in job.error and "contracts/hack.py" in job.error


def test_failing_check_fails_job_and_discard_cleans_up(tmp_path, repo):
    h = make_harness(tmp_path, repo, checks=[Check(name="never", cmd="exit 1")])
    job = h.run(h.create("demo", {"name": "x"}).id)
    assert job.state == "failed" and "never" in job.error
    with pytest.raises(HarnessError):
        h.push(job.id)
    out = h.discard(job.id)
    assert out.state == "discarded"
    branches = subprocess.run(["git", "branch", "--list", job.branch], cwd=repo, capture_output=True, text=True).stdout
    assert branches.strip() == "" and not (tmp_path / "wts" / job.id).exists()


def test_unknown_task_and_agent(tmp_path, repo):
    h = make_harness(tmp_path, repo)
    with pytest.raises(HarnessError):
        h.create("nope", {})
    with pytest.raises(KeyError):
        h.create("demo", {"name": "x"}, agent="missing")
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `uv run pytest -q tests/harness/test_runner.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.agents'`

- [ ] **Step 4: Create `harness/gitops.py`**

```python
"""Git operations for harness jobs (worktree per job, commit, diff, push, cleanup)."""
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: str | Path) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout


def create_worktree(repo: Path, path: Path, branch: str, base_ref: str, fetch: bool = True,
                    remote: str = "origin") -> str:
    """Create `path` on new `branch` from `base_ref`; return the base commit sha."""
    if fetch:
        git(["fetch", "-q", remote], repo)
    base_sha = git(["rev-parse", base_ref], repo).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "-q", "-b", branch, str(path), base_sha], repo)
    return base_sha


def stage_all(worktree: Path) -> list[str]:
    """`git add -A` and return the staged paths (repo-relative, posix)."""
    git(["add", "-A"], worktree)
    out = git(["diff", "--cached", "--name-only"], worktree)
    return [line for line in out.splitlines() if line]


def commit(worktree: Path, message: str) -> str:
    git(["commit", "-q", "-m", message], worktree)
    return git(["rev-parse", "HEAD"], worktree).strip()


def diff(worktree: Path, base_sha: str, stat: bool = False) -> str:
    return git(["diff", "--stat" if stat else "--patch", f"{base_sha}..HEAD"], worktree)


def push(worktree: Path, branch: str, remote: str = "origin") -> None:
    git(["push", "-q", "-u", remote, branch], worktree)


def remove_worktree(repo: Path, path: Path, branch: str, delete_branch: bool = True) -> None:
    if path.exists():
        git(["worktree", "remove", "--force", str(path)], repo)
    git(["worktree", "prune"], repo)
    if delete_branch:
        try:
            git(["branch", "-D", branch], repo)
        except GitError:
            pass
```

- [ ] **Step 5: Create `harness/agents.py`**

```python
"""Pluggable coding-agent CLIs (configs/agents.yaml)."""
from pathlib import Path
import yaml
from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    argv: list[str]           # placeholders: {workdir} {prompt} {prompt_file}
    description: str = ""


class AgentsConfig(BaseModel):
    default: str
    agents: dict[str, AgentConfig]

    def get(self, name: str | None) -> AgentConfig:
        key = name or self.default
        if key not in self.agents:
            raise KeyError(f"unknown agent '{key}'; configured: {sorted(self.agents)}")
        return self.agents[key]


def load_agents(path: str | Path) -> AgentsConfig:
    raw = yaml.safe_load(Path(path).read_text())
    agents = {k: AgentConfig(name=k, **v) for k, v in raw["agents"].items()}
    return AgentsConfig(default=raw["default"], agents=agents)


def build_argv(agent: AgentConfig, workdir: Path, prompt: str, prompt_file: Path) -> list[str]:
    values = {"workdir": str(workdir), "prompt": prompt, "prompt_file": str(prompt_file)}
    return [a.format(**values) for a in agent.argv]
```

- [ ] **Step 6: Create `configs/agents.yaml`**

```yaml
# Coding-agent CLIs the harness can drive. The prompt is passed as {prompt} (and also saved to {prompt_file}).
# The agent always runs inside the job's git worktree ({workdir}), never in your working folder.
default: copilot
agents:
  copilot:
    description: "GitHub Copilot CLI (headless). Needs `copilot` logged in on the host."
    argv: [copilot, -C, "{workdir}", -p, "{prompt}", --allow-all-tools, --deny-tool, "shell(git push)",
           --deny-tool, "shell(git commit)", --output-format, json]
  claude:
    description: "Claude Code (optional). Needs `claude` installed and logged in."
    argv: [bash, -c, 'cd "$0" && claude -p "$(cat "$1")" --dangerously-skip-permissions --output-format stream-json --verbose',
           "{workdir}", "{prompt_file}"]
  codex:
    description: "OpenAI Codex CLI (optional). Needs `codex` installed and logged in."
    argv: [codex, exec, --full-auto, -C, "{workdir}", "{prompt}"]
```

- [ ] **Step 7: Create `harness/runner.py`**

```python
"""Harness job lifecycle: worktree -> agent -> path guard -> acceptance checks -> commit -> review -> push/discard.

State order: created, worktree, agent_running, guard, checks, ready_for_review, then pushed | discarded,
or failed (with `error`) at any step.
"""
import threading
import time
import uuid
from pathlib import Path
from pydantic import BaseModel, Field
from harness import gitops
from harness.agents import AgentsConfig, build_argv
from harness.guard import path_violations
from harness.model import TaskTemplate, allowed_for, checks_for, render_prompt, validate_params
from orchestrator.process import read_log, start_process


class HarnessSettings(BaseModel):
    repo: Path
    var_dir: Path                 # job records + logs: <var_dir>/agent/<id>/
    worktrees_dir: Path           # job worktrees: <worktrees_dir>/<id>/
    base_ref: str = "origin/main"
    remote: str = "origin"
    fetch: bool = True


class CheckResult(BaseModel):
    name: str
    ok: bool
    state: str


class AgentJob(BaseModel):
    id: str
    task: str
    title: str
    params: dict
    extra: str = ""
    agent: str
    branch: str
    worktree: str
    state: str = "created"
    error: str = ""
    base_sha: str = ""
    head_sha: str = ""
    changed_files: list[str] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    created: float
    updated: float


class HarnessError(RuntimeError):
    pass


class Harness:
    def __init__(self, settings: HarnessSettings, templates: dict[str, TaskTemplate], agents: AgentsConfig):
        self.s, self.templates, self.agents = settings, templates, agents
        self._lock = threading.Lock()

    # persistence ---------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.s.var_dir / "agent" / job_id

    def _save(self, job: AgentJob) -> AgentJob:
        job.updated = time.time()
        d = self.job_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(job.model_dump_json(indent=2))
        return job

    def get(self, job_id: str) -> AgentJob:
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            raise KeyError(job_id)
        return AgentJob.model_validate_json(path.read_text())

    def list_jobs(self) -> list[AgentJob]:
        root = self.s.var_dir / "agent"
        jobs = [AgentJob.model_validate_json(p.read_text()) for p in root.glob("*/job.json")] if root.exists() else []
        return sorted(jobs, key=lambda j: j.created, reverse=True)

    def log(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        return read_log(self.job_dir(job_id) / "run.log", offset)

    def _note(self, job: AgentJob, text: str) -> None:
        with open(self.job_dir(job.id) / "run.log", "a") as f:
            f.write(f"\n=== [harness] {text}\n")

    # lifecycle -----------------------------------------------------------
    def create(self, task_id: str, params: dict, extra: str = "", agent: str | None = None) -> AgentJob:
        if task_id not in self.templates:
            raise HarnessError(f"unknown task '{task_id}'")
        template = self.templates[task_id]
        clean = validate_params(template, params)
        agent_cfg = self.agents.get(agent)
        job_id = f"{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        now = time.time()
        job = AgentJob(id=job_id, task=task_id, title=template.title, params=clean, extra=extra,
                       agent=agent_cfg.name, branch=f"agent/{task_id}-{job_id}",
                       worktree=str(self.s.worktrees_dir / job_id), created=now, updated=now)
        self._save(job)
        prompt = render_prompt(template, clean, extra)
        (self.job_dir(job_id) / "prompt.md").write_text(prompt)
        return job

    def _fail(self, job: AgentJob, error: str) -> AgentJob:
        job.state, job.error = "failed", error
        self._note(job, f"FAILED: {error}")
        return self._save(job)

    def run(self, job_id: str) -> AgentJob:
        """Execute the job synchronously up to `ready_for_review` (or `failed`)."""
        job = self.get(job_id)
        template = self.templates[job.task]
        wt = Path(job.worktree)
        d = self.job_dir(job_id)
        try:
            job.state = "worktree"
            self._save(job)
            self._note(job, f"worktree {wt} on {job.branch} from {self.s.base_ref}")
            job.base_sha = gitops.create_worktree(self.s.repo, wt, job.branch, self.s.base_ref,
                                                  fetch=self.s.fetch, remote=self.s.remote)

            job.state = "agent_running"
            self._save(job)
            prompt_file = d / "prompt.md"
            argv = build_argv(self.agents.get(job.agent), wt, prompt_file.read_text(), prompt_file)
            self._note(job, f"agent {job.agent}: {argv[0]} ...")
            state = start_process(argv, d / "run.log", cwd=wt, timeout_s=template.timeout_min * 60,
                                  append=True).wait()
            if state != "done":
                return self._fail(job, f"agent process ended with state '{state}'")

            job.state = "guard"
            job.changed_files = gitops.stage_all(wt)
            self._save(job)
            if not job.changed_files:
                return self._fail(job, "agent made no changes")
            bad = path_violations(job.changed_files, allowed_for(template, job.params))
            if bad:
                return self._fail(job, f"changes outside allowed paths: {bad}")

            job.state = "checks"
            self._save(job)
            for check in checks_for(template, job.params):
                self._note(job, f"check '{check.name}': {check.cmd}")
                st = start_process(check.cmd, d / "run.log", cwd=wt, timeout_s=check.timeout_s, append=True).wait()
                job.checks.append(CheckResult(name=check.name, ok=st == "done", state=st))
                self._save(job)
            failed = [c.name for c in job.checks if not c.ok]
            if failed:
                return self._fail(job, f"acceptance checks failed: {failed}")

            job.head_sha = gitops.commit(wt, f"agent({job.task}): {job.title} [{job.id}]")
            (d / "diff.patch").write_text(gitops.diff(wt, job.base_sha))
            job.state = "ready_for_review"
            self._note(job, "ready for review")
            return self._save(job)
        except Exception as e:  # noqa: BLE001 - every failure must land in the job record
            return self._fail(job, f"{type(e).__name__}: {e}")

    def start(self, job_id: str) -> threading.Thread:
        t = threading.Thread(target=self.run, args=(job_id,), daemon=True)
        t.start()
        return t

    def diff(self, job_id: str) -> str:
        p = self.job_dir(job_id) / "diff.patch"
        return p.read_text() if p.exists() else ""

    def push(self, job_id: str) -> AgentJob:
        job = self.get(job_id)
        if job.state != "ready_for_review":
            raise HarnessError(f"job is '{job.state}', only 'ready_for_review' can be pushed")
        gitops.push(Path(job.worktree), job.branch, self.s.remote)
        gitops.remove_worktree(self.s.repo, Path(job.worktree), job.branch, delete_branch=False)
        job.state = "pushed"
        self._note(job, f"pushed {job.branch}")
        return self._save(job)

    def discard(self, job_id: str) -> AgentJob:
        job = self.get(job_id)
        if job.state in ("agent_running", "checks", "worktree", "guard"):
            raise HarnessError(f"job is still '{job.state}'")
        gitops.remove_worktree(self.s.repo, Path(job.worktree), job.branch, delete_branch=True)
        job.state = "discarded"
        self._note(job, "discarded")
        return self._save(job)
```

- [ ] **Step 8: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/harness/test_runner.py`
Expected: 4 passed (uses a throwaway git repo with a bare `origin` and a fake agent; no network)

- [ ] **Step 9: Commit**

```bash
git add harness configs/agents.yaml tests/harness
git commit -m "feat(harness): worktree-isolated agent jobs with path guard, checks and review gate"
```

### Task 7: Harness HTTP API, agent worker, factory and CLI

**Files:**
- Create: `harness/api.py`, `harness/server.py`, `harness/factory.py`, `harness/__main__.py`
- Test: `tests/harness/test_api.py`

**Interfaces:**
- Consumes: Task 6 `Harness`, `HarnessError`; Task 5 `ParamError`, `load_templates`; `load_agents`.
- Produces:
  - `create_agent_router(harness) -> APIRouter` (mounted at `/api/agent`): `GET /tasks`, `GET /agents`, `POST /jobs` `{task, params, extra, agent, start=True}`, `GET /jobs`, `GET /jobs/{id}`, `GET /jobs/{id}/log?offset=`, `GET /jobs/{id}/prompt`, `GET /jobs/{id}/diff`, `POST /jobs/{id}/push`, `POST /jobs/{id}/discard`. Errors: 400 for bad input, 404 for an unknown job, 409 for a wrong state.
  - `create_agent_app(harness) -> FastAPI` with `/health`.
  - `default_harness(repo_root, var_dir=None, fetch=True) -> Harness`. Worktrees go in `$CATHLAB_WORKTREES` or `<repo parent>/.<repo name>-agent-worktrees`; the base is `$CATHLAB_AGENT_BASE` or `origin/main`.
  - CLI: `python -m harness list | run <task> --param k=v [--extra] [--agent] | show | diff | push | discard <id> | serve --port 8790`.

- [ ] **Step 1: Write the failing test `tests/harness/test_api.py`**

```python
import time
from fastapi.testclient import TestClient
from harness.server import create_agent_app
from tests.harness.conftest import make_harness


def wait(c, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/api/agent/jobs/{job_id}").json()["state"]
        if st in ("ready_for_review", "failed"):
            return st
        time.sleep(0.05)
    return st


def test_agent_api_end_to_end(tmp_path, repo):
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    assert c.get("/health").json()["capabilities"] == ["harness"]
    assert [t["id"] for t in c.get("/api/agent/tasks").json()] == ["demo"]
    assert c.get("/api/agent/agents").json()["default"] == "fake"

    bad = c.post("/api/agent/jobs", json={"task": "demo", "params": {}})
    assert bad.status_code == 400 and "missing required" in bad.json()["detail"]

    job = c.post("/api/agent/jobs", json={"task": "demo", "params": {"name": "hoop"}}).json()
    assert wait(c, job["id"]) == "ready_for_review"
    assert "Create hoop" in c.get(f"/api/agent/jobs/{job['id']}/prompt").text
    assert "made by fake agent" in c.get(f"/api/agent/jobs/{job['id']}/diff").text
    log = c.get(f"/api/agent/jobs/{job['id']}/log").json()
    assert "ready for review" in log["text"] and log["offset"] > 0
    assert c.post(f"/api/agent/jobs/{job['id']}/push").json()["state"] == "pushed"
    assert c.post(f"/api/agent/jobs/{job['id']}/push").status_code == 409
    assert c.get("/api/agent/jobs/unknown").status_code == 404
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/harness/test_api.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.server'`

- [ ] **Step 3: Create `harness/api.py`**

```python
"""HTTP API of the agent harness (mounted at /api/agent by the control plane or the agent worker)."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from harness.model import ParamError
from harness.runner import Harness, HarnessError


class JobCreate(BaseModel):
    task: str
    params: dict = Field(default_factory=dict)
    extra: str = ""
    agent: str | None = None
    start: bool = True


def create_agent_router(harness: Harness) -> APIRouter:
    r = APIRouter()

    def job_or_404(job_id: str):
        try:
            return harness.get(job_id)
        except KeyError:
            raise HTTPException(404, f"unknown agent job {job_id}") from None

    @r.get("/tasks")
    def tasks() -> list[dict]:
        return [t.model_dump() for t in harness.templates.values()]

    @r.get("/agents")
    def agents() -> dict:
        return {"default": harness.agents.default,
                "agents": [{"name": a.name, "description": a.description} for a in harness.agents.agents.values()]}

    @r.post("/jobs")
    def create(req: JobCreate) -> dict:
        try:
            job = harness.create(req.task, req.params, req.extra, req.agent)
        except (HarnessError, ParamError, KeyError) as e:
            raise HTTPException(400, str(e)) from e
        if req.start:
            harness.start(job.id)
        return job.model_dump()

    @r.get("/jobs")
    def list_jobs() -> list[dict]:
        return [j.model_dump() for j in harness.list_jobs()]

    @r.get("/jobs/{job_id}")
    def get(job_id: str) -> dict:
        return job_or_404(job_id).model_dump()

    @r.get("/jobs/{job_id}/log")
    def log(job_id: str, offset: int = 0) -> dict:
        job = job_or_404(job_id)
        text, new_offset = harness.log(job_id, offset)
        return {"text": text, "offset": new_offset, "state": job.state}

    @r.get("/jobs/{job_id}/prompt", response_class=PlainTextResponse)
    def prompt(job_id: str) -> str:
        job_or_404(job_id)
        return (harness.job_dir(job_id) / "prompt.md").read_text()

    @r.get("/jobs/{job_id}/diff", response_class=PlainTextResponse)
    def diff(job_id: str) -> str:
        job_or_404(job_id)
        return harness.diff(job_id)

    @r.post("/jobs/{job_id}/push")
    def push(job_id: str) -> dict:
        job_or_404(job_id)
        try:
            return harness.push(job_id).model_dump()
        except HarnessError as e:
            raise HTTPException(409, str(e)) from e

    @r.post("/jobs/{job_id}/discard")
    def discard(job_id: str) -> dict:
        job_or_404(job_id)
        try:
            return harness.discard(job_id).model_dump()
        except HarnessError as e:
            raise HTTPException(409, str(e)) from e

    return r
```

- [ ] **Step 4: Create `harness/server.py`**

```python
"""Standalone agent worker: the harness API on its own port. Runs on the host (needs the agent CLI login + git)."""
from fastapi import FastAPI
from harness.api import create_agent_router
from harness.runner import Harness


def create_agent_app(harness: Harness) -> FastAPI:
    app = FastAPI(title="cathlab agent worker")
    app.include_router(create_agent_router(harness), prefix="/api/agent")

    @app.get("/health")
    def health() -> dict:
        return {"name": "agent", "capabilities": ["harness"], "tasks": sorted(harness.templates)}

    return app
```

- [ ] **Step 5: Create `harness/factory.py`**

```python
"""Build the default Harness for a repo checkout."""
import os
from pathlib import Path
from harness.agents import load_agents
from harness.model import load_templates
from harness.runner import Harness, HarnessSettings

PACKAGE_DIR = Path(__file__).resolve().parent


def default_harness(repo_root: str | Path, var_dir: str | Path | None = None, fetch: bool = True) -> Harness:
    repo_root = Path(repo_root).resolve()
    var = Path(var_dir) if var_dir else repo_root / "var"
    worktrees = Path(os.environ.get("CATHLAB_WORKTREES", repo_root.parent / f".{repo_root.name}-agent-worktrees"))
    settings = HarnessSettings(repo=repo_root, var_dir=var, worktrees_dir=worktrees,
                               base_ref=os.environ.get("CATHLAB_AGENT_BASE", "origin/main"), fetch=fetch)
    return Harness(settings, load_templates(PACKAGE_DIR / "tasks"), load_agents(repo_root / "configs/agents.yaml"))
```

- [ ] **Step 6: Create `harness/__main__.py`**

```python
"""Agent harness CLI.

  python -m harness list
  python -m harness run add_object --param name=guidewire_hoop --param shape=cylinder [--extra "..."] [--agent copilot]
  python -m harness show <job_id> | diff <job_id> | push <job_id> | discard <job_id>
  python -m harness serve --port 8790        # standalone agent worker (HTTP API at /api/agent)
"""
import argparse
import sys
from pathlib import Path
from harness.factory import default_harness

ROOT = Path(__file__).resolve().parents[1]


def parse_params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got '{pair}'")
        k, v = pair.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--no-fetch", action="store_true", help="don't `git fetch` before creating the worktree")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("task")
    run.add_argument("--param", action="append", default=[])
    run.add_argument("--extra", default="")
    run.add_argument("--agent")
    for name in ("show", "diff", "push", "discard"):
        sub.add_parser(name).add_argument("job_id")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)

    h = default_harness(args.repo, fetch=not args.no_fetch)
    if args.cmd == "list":
        for t in h.templates.values():
            params = ", ".join(p.name for p in t.params)
            print(f"{t.id:22s} {t.title}  [{params}]")
    elif args.cmd == "run":
        job = h.create(args.task, parse_params(args.param), args.extra, args.agent)
        print(f"job {job.id} on branch {job.branch}\nlog: {h.job_dir(job.id) / 'run.log'}")
        job = h.run(job.id)
        print(f"state: {job.state}" + (f"\nerror: {job.error}" if job.error else ""))
        if job.state == "ready_for_review":
            print(f"review: python -m harness diff {job.id}\npush:   python -m harness push {job.id}")
        return 0 if job.state == "ready_for_review" else 1
    elif args.cmd == "show":
        print(h.get(args.job_id).model_dump_json(indent=2))
    elif args.cmd == "diff":
        sys.stdout.write(h.diff(args.job_id))
    elif args.cmd == "push":
        print(h.push(args.job_id).state)
    elif args.cmd == "discard":
        print(h.discard(args.job_id).state)
    elif args.cmd == "serve":
        import uvicorn
        from harness.server import create_agent_app
        uvicorn.run(create_agent_app(h), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/harness/test_api.py`
Expected: 1 passed

- [ ] **Step 8: Commit**

```bash
git add harness tests/harness/test_api.py
git commit -m "feat(harness): HTTP API, standalone agent worker and CLI"
```

### Task 8: Control plane API

**Files:**
- Create: `orchestrator/control.py`, `configs/workers.yaml`
- Test: `tests/orchestrator/test_control.py`

**Interfaces:**
- Consumes: Task 1 `load_catalog`, `selection_problems`; Task 2 `Store`; Task 4 `create_worker_app` (in tests); Task 7 `create_agent_router`, `default_harness`.
- Produces: `create_control_app(repo_root, var_dir=None, catalog_path=None, workers_path=None, frontend_dir=None, harness=None) -> FastAPI` with:
  - `GET /api/health`, `GET /api/catalog`;
  - `POST /api/selection/check` `{selection}` → `{problems}`;
  - `POST /api/runs` `{selection, note}` → Run, or 409 `{message, problems}`; the run is frozen to `var/runs/<id>/run.yaml`;
  - `GET /api/runs`, `GET /api/runs/{id}`, `GET /api/jobs?kind=&run_id=`, `GET /api/jobs/{id}`;
  - `GET /api/workers` (health of each `configs/workers.yaml` entry; `$CATHLAB_WORKERS` overrides the file);
  - `/api/agent/*` when `harness` is given;
  - static `frontend/` at `/`;
  - `app.state.store`.
- CLI: `python -m orchestrator.control --port 8700 [--host] [--no-agent]`.

- [ ] **Step 1: Write the failing test `tests/orchestrator/test_control.py`**

```python
import threading
from pathlib import Path
import uvicorn
from fastapi.testclient import TestClient
from orchestrator.control import create_control_app
from orchestrator.worker import create_worker_app

ROOT = Path(__file__).resolve().parents[2]


def client(tmp_path, workers_yaml: str = "workers: []\n", frontend=True, harness=None):
    wp = tmp_path / "workers.yaml"
    wp.write_text(workers_yaml)
    fe = tmp_path / "fe"
    if frontend:
        fe.mkdir()
        (fe / "index.html").write_text("<title>CathLab Control Station</title>")
    app = create_control_app(ROOT, var_dir=tmp_path / "var", workers_path=wp, frontend_dir=fe, harness=harness)
    return TestClient(app)


def test_catalog_and_selection_check(tmp_path):
    c = client(tmp_path)
    cat = c.get("/api/catalog").json()
    assert "simulator" in cat["sections"] and cat["defaults"]["model"] == "smolvla"
    ok = c.post("/api/selection/check", json={"selection": cat["defaults"]}).json()
    assert ok["problems"] == []
    bad = c.post("/api/selection/check", json={"selection": {**cat["defaults"], "simulator": "isaac"}}).json()
    assert any(p["harness_task"] == "add_sim_backend" for p in bad["problems"])


def test_runs_are_frozen_to_disk_and_blocked_when_not_runnable(tmp_path):
    c = client(tmp_path)
    defaults = c.get("/api/catalog").json()["defaults"]
    run = c.post("/api/runs", json={"selection": defaults, "note": "smoke"}).json()
    assert (tmp_path / "var/runs" / run["id"] / "run.yaml").exists()
    assert c.get(f"/api/runs/{run['id']}").json()["note"] == "smoke"
    assert [r["id"] for r in c.get("/api/runs").json()] == [run["id"]]
    r = c.post("/api/runs", json={"selection": {**defaults, "expert": "graspgen"}})
    assert r.status_code == 409 and r.json()["detail"]["problems"][0]["harness_task"] == "add_expert"
    assert c.get("/api/runs/nope").status_code == 404


def test_jobs_endpoints(tmp_path):
    c = client(tmp_path)
    store = c.app.state.store
    j = store.create_job("stage", "generate", run_id="run-x")
    assert [x["id"] for x in c.get("/api/jobs", params={"run_id": "run-x"}).json()] == [j.id]
    assert c.get(f"/api/jobs/{j.id}").json()["name"] == "generate"
    assert c.get("/api/jobs/nope").status_code == 404


def test_workers_health_online_and_offline(tmp_path):
    app = create_worker_app("robosuite", ["generate"], tmp_path / "w")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8799, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        pass
    try:
        c = client(tmp_path, "workers:\n  - {name: robosuite, url: 'http://127.0.0.1:8799'}\n"
                             "  - {name: lerobot, url: 'http://127.0.0.1:8798'}\n")
        ws = {w["name"]: w for w in c.get("/api/workers").json()}
        assert ws["robosuite"]["healthy"] and ws["robosuite"]["capabilities"] == ["generate"]
        assert not ws["lerobot"]["healthy"] and ws["lerobot"]["error"]
    finally:
        server.should_exit = True
        t.join(5)


def test_frontend_served_and_agent_optional(tmp_path):
    c = client(tmp_path)
    assert "CathLab Control Station" in c.get("/").text
    assert c.get("/api/health").json()["agent"] is False
    assert c.get("/api/agent/tasks").status_code in (404, 405)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/orchestrator/test_control.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.control'`

- [ ] **Step 3: Create `orchestrator/control.py`**

```python
"""Control plane: catalog, runs, jobs, worker health, agent harness, and the UI.

Start (dev): python -m orchestrator.control --port 8700   ->  open http://localhost:8700
"""
import argparse
import os
from pathlib import Path
import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from orchestrator.catalog import load_catalog, selection_problems
from orchestrator.store import Store

ROOT = Path(__file__).resolve().parents[1]


class SelectionBody(BaseModel):
    selection: dict
    note: str = ""


class WorkerEntry(BaseModel):
    name: str
    url: str


class WorkersConfig(BaseModel):
    workers: list[WorkerEntry] = Field(default_factory=list)


def load_workers(path: str | Path) -> WorkersConfig:
    p = Path(path)
    return WorkersConfig.model_validate(yaml.safe_load(p.read_text())) if p.exists() else WorkersConfig()


def worker_health(entry: WorkerEntry, timeout: float = 1.0) -> dict:
    try:
        r = httpx.get(entry.url.rstrip("/") + "/health", timeout=timeout)
        r.raise_for_status()
        return {"name": entry.name, "url": entry.url, "healthy": True, **r.json(), "error": ""}
    except Exception as e:  # noqa: BLE001 - any failure means "offline"
        return {"name": entry.name, "url": entry.url, "healthy": False, "capabilities": [], "error": str(e)[:200]}


def create_control_app(repo_root: str | Path = ROOT, var_dir: str | Path | None = None,
                       catalog_path: str | Path | None = None, workers_path: str | Path | None = None,
                       frontend_dir: str | Path | None = None, harness=None) -> FastAPI:
    repo_root = Path(repo_root)
    var_dir = Path(var_dir) if var_dir else repo_root / "var"
    catalog_path = Path(catalog_path or repo_root / "configs/catalog.yaml")
    workers_path = Path(workers_path or os.environ.get("CATHLAB_WORKERS", repo_root / "configs/workers.yaml"))
    frontend_dir = Path(frontend_dir or repo_root / "frontend")
    store = Store(var_dir / "control.db")
    app = FastAPI(title="CathLab Control Station")
    app.state.store = store

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "repo": str(repo_root), "agent": harness is not None}

    @app.get("/api/catalog")
    def catalog() -> dict:
        return load_catalog(catalog_path).model_dump()

    @app.post("/api/selection/check")
    def check(body: SelectionBody) -> dict:
        return {"problems": [p.model_dump() for p in selection_problems(load_catalog(catalog_path), body.selection)]}

    @app.post("/api/runs")
    def create_run(body: SelectionBody) -> dict:
        problems = selection_problems(load_catalog(catalog_path), body.selection)
        if problems:
            raise HTTPException(409, {"message": "selection is not runnable", "problems": [p.model_dump() for p in problems]})
        run = store.create_run(body.selection, body.note)
        run_dir = var_dir / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run.model_dump(), sort_keys=False))
        return run.model_dump()

    @app.get("/api/runs")
    def runs() -> list[dict]:
        return [r.model_dump() for r in store.list_runs()]

    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return r.model_dump()

    @app.get("/api/jobs")
    def jobs(kind: str | None = None, run_id: str | None = None) -> list[dict]:
        return [j.model_dump() for j in store.list_jobs(kind=kind, run_id=run_id)]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        j = store.get_job(job_id)
        if j is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return j.model_dump()

    @app.get("/api/workers")
    def workers() -> list[dict]:
        return [worker_health(w) for w in load_workers(workers_path).workers]

    if harness is not None:
        from harness.api import create_agent_router
        app.include_router(create_agent_router(harness), prefix="/api/agent")

    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--no-agent", action="store_true", help="don't mount the agent harness (e.g. inside Docker)")
    args = ap.parse_args()
    harness = None
    if not args.no_agent:
        from harness.factory import default_harness
        harness = default_harness(ROOT)
    uvicorn.run(create_control_app(ROOT, harness=harness), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `configs/workers.yaml`**

```yaml
# Workers the control plane talks to (dev defaults: native processes on this machine).
# Docker / remote GPU: point CATHLAB_WORKERS at another file, or edit the URLs.
workers:
  - {name: robosuite, url: "http://127.0.0.1:8711"}
  - {name: lerobot, url: "http://127.0.0.1:8712"}
  - {name: isaac, url: "http://127.0.0.1:8713"}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator/test_control.py`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add orchestrator/control.py configs/workers.yaml tests/orchestrator/test_control.py
git commit -m "feat(orchestrator): control plane API (catalog, runs, jobs, workers, agent)"
```

### Task 9: Starter task templates

**Files:**
- Create: `harness/tasks/add_object.yaml`, `extend_dataset.yaml`, `add_variation_axis.yaml`, `add_sim_backend.yaml`, `add_policy_adapter.yaml`, `add_expert.yaml`, `free_task.yaml`, `scripts/assert_expert_rate.py`
- Test: `tests/harness/test_templates.py`

**Interfaces:**
- Consumes: Task 5 `load_templates`, `validate_params`, `render_prompt`, `allowed_for`, `checks_for`; Task 6 `load_agents`; Task 1 `load_catalog`.
- Produces:
  - 7 templates. Every catalog `harness_task` and section `add_task` must name one of them.
  - `scripts/assert_expert_rate.py <summary.json> --min 0.8` exits 1 if any object is below the rate. It reads `per_object.<obj>.rate|success|attempted` from `scripts/robosuite_expert_check.py --json`.

- [ ] **Step 1: Write the failing test `tests/harness/test_templates.py`**

```python
import json
from pathlib import Path
import pytest
import yaml
from harness.agents import load_agents
from harness.model import allowed_for, checks_for, load_templates, render_prompt, validate_params
from orchestrator.catalog import load_catalog

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = load_templates(ROOT / "harness/tasks")
EXPECTED = {"add_object", "extend_dataset", "add_variation_axis", "add_sim_backend", "add_policy_adapter",
            "add_expert", "free_task"}

SAMPLE = {
    "add_object": {"name": "hoop", "label": "Hoop", "radius": 0.06, "length": 0.02, "mass": 0.05},
    "extend_dataset": {"out": "/tmp/x"},
    "add_variation_axis": {"axis_id": "camera_jitter", "label": "Camera jitter", "description": "±2 cm"},
    "add_sim_backend": {"sim": "mujoco"},
    "add_policy_adapter": {"model_id": "pi05"},
    "add_expert": {},
    "free_task": {"goal": "fix it", "paths": "eval/, tests/eval/"},
}


def test_all_starter_templates_present():
    assert set(TEMPLATES) == EXPECTED


@pytest.mark.parametrize("task_id", sorted(EXPECTED))
def test_template_renders_with_sample_params(task_id):
    t = TEMPLATES[task_id]
    params = validate_params(t, SAMPLE[task_id])
    prompt = render_prompt(t, params)
    assert "{" not in "".join(allowed_for(t, params)) and "{" not in " ".join(c.cmd for c in checks_for(t, params))
    assert t.title in prompt and allowed_for(t, params)


def test_catalog_harness_tasks_exist():
    cat = load_catalog(ROOT / "configs/catalog.yaml")
    refs = {i.harness_task for s in cat.sections.values() for i in s.items if i.harness_task}
    refs |= {s.add_task for s in cat.sections.values() if s.add_task}
    assert refs <= set(TEMPLATES)


def test_agents_config_has_copilot_default():
    agents = load_agents(ROOT / "configs/agents.yaml")
    assert agents.default == "copilot" and {"copilot", "claude", "codex"} <= set(agents.agents)
    assert "{prompt}" in agents.get(None).argv and "{workdir}" in agents.get(None).argv


def test_assert_expert_rate(tmp_path):
    from scripts.assert_expert_rate import main
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"per_object": {"a": {"rate": 1.0, "success": 5, "attempted": 5},
                                            "b": {"rate": 0.6, "success": 3, "attempted": 5}}}))
    assert main([str(p), "--min", "0.5"]) == 0
    assert main([str(p), "--min", "0.8"]) == 1
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest -q tests/harness/test_templates.py`
Expected: FAIL with `AssertionError (no templates found) on test_all_starter_templates_present`

- [ ] **Step 3: Create `harness/tasks/add_object.yaml`**

```yaml
id: add_object
title: Add an object to the robosuite scene
description: >
  Add a new pickable object (geometry, mass, grasp type, colours) to a NEW versioned variation spec, the robosuite
  scene builder, the task config and the catalog, so the scripted expert can pick it and episodes can be generated.
catalog_target: objects
params:
  - {name: name, type: str, help: "object id, snake_case, e.g. guidewire_hoop"}
  - {name: label, type: str, help: "UI label, e.g. 'Guidewire hoop (Ø120 mm)'"}
  - {name: shape, type: choice, choices: [cylinder, syringe, box], default: cylinder, help: "box needs a new shape in ObjectSpec + env.py"}
  - {name: radius, type: float, help: "metres (box: half width)"}
  - {name: length, type: float, help: "metres (height if upright, long axis if lying)"}
  - {name: upright, type: bool, default: true}
  - {name: mass, type: float, help: "kg"}
  - {name: grasp, type: choice, choices: [top, side], default: top}
  - {name: spec_out, type: str, default: cathlab_v2, help: "new variation spec file name (never edit cathlab_v1)"}
context:
  - AGENTS.md
  - docs/specs/2026-10-22-sim-framework-design.md
  - configs/variation/cathlab_v1.yaml
  - dataengine/variation.py
  - dataengine/expert.py
  - sim/robosuite/env.py
  - sim/robosuite/randomize.py
  - configs/tasks/cathlab_pick.yaml
  - configs/catalog.yaml
  - sim/robosuite/README.md
prompt: |
  Add the object `{name}` ("{label}") to the robosuite pick task.
  1. Create configs/variation/{spec_out}.yaml as a copy of configs/variation/cathlab_v1.yaml with variation_id `{spec_out}`
     and a new `objects.{name}` entry: shape {shape}, radius {radius}, length {length}, upright {upright}, mass {mass}, grasp {grasp}.
     Add an RGBA colour for `{name}` to EVERY palette in `obj_colors` (5 palettes, visibly distinct per palette).
     Do NOT edit cathlab_v1.yaml: the published v1 dataset depends on it.
  2. If shape is `box` (or anything ObjectSpec/env.py does not support yet), extend `ObjectSpec.shape` in dataengine/variation.py
     and the object builder in sim/robosuite/env.py, keeping existing shapes unchanged.
  3. Add `{name}` to `objects` in configs/tasks/cathlab_pick.yaml and an item {{id: {name}, label: "{label}", status: ready}}
     to the `objects` section of configs/catalog.yaml.
  4. Make sure layouts still separate all objects (min_separation, reach corridor) and the expert grasp template suits the
     object; tune `sim/robosuite/params.py` only if needed and explain why in docs/decisions.md (one new table row).
  5. Add or extend a test under tests/ that loads configs/variation/{spec_out}.yaml and checks the new object exists in every palette.
allowed_paths:
  - configs/variation/
  - configs/tasks/cathlab_pick.yaml
  - configs/catalog.yaml
  - dataengine/variation.py
  - sim/robosuite/
  - tests/
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
  - {name: robosuite-tests, cmd: "conda run -n robosuite --no-capture-output env MUJOCO_GL=egl python -m pytest -q tests/sim"}
  - name: expert-rate
    cmd: >-
      conda run -n robosuite --no-capture-output env MUJOCO_GL=egl python scripts/robosuite_expert_check.py
      --n 5 --objects {name} --spec configs/variation/{spec_out}.yaml --json var/expert_check.json
      && python3 scripts/assert_expert_rate.py var/expert_check.json --min 0.8
    timeout_s: 1800
timeout_min: 45
```

- [ ] **Step 4: Create `harness/tasks/extend_dataset.yaml`**

```yaml
id: extend_dataset
title: Generate more episodes and publish them
description: >
  Run the scripted expert in robosuite for a window of variation configs, convert to LeRobot v3, preview, optionally
  push to the Hugging Face Hub, and log the result. Data lands OUTSIDE git under the given output folder.
catalog_target: data
params:
  - {name: spec, type: str, default: cathlab_v1, help: "variation spec name in configs/variation/"}
  - {name: split, type: choice, choices: [train, in_distribution, held_out_layout, held_out_obj_color, held_out_table_color, all_held_out], default: train}
  - {name: start, type: int, default: 0}
  - {name: n, type: int, default: 24, help: "configs in the window"}
  - {name: num_shards, type: int, default: 4}
  - {name: out, type: str, help: "absolute output folder outside the repo, e.g. /home/me/cathlab-data/pick_ext_v2"}
  - {name: dataset_id, type: str, default: cathlab/pick_robosuite}
  - {name: version, type: str, default: v2}
  - {name: hf_repo, type: str, required: false, help: "e.g. <org>/pick_robosuite; empty = don't upload"}
context:
  - AGENTS.md
  - docs/tasks/robosuite-collection.md
  - scripts/robosuite_generate.py
  - scripts/npz_to_lerobot.py
  - scripts/preview_episodes.py
  - docs/data.md
prompt: |
  Produce episodes {start}..{start}+{n} of split `{split}` from configs/variation/{spec}.yaml, following
  docs/tasks/robosuite-collection.md exactly:
  1. Run {num_shards} shards of scripts/robosuite_generate.py in parallel in the `robosuite` conda env with MUJOCO_GL=egl,
     `--spec configs/variation/{spec}.yaml --split {split} --start {start} --n {n} --num-shards {num_shards} --out {out}
     --dataset-id {dataset_id} --version {version}`. Wait for all shards.
  2. Convert with scripts/npz_to_lerobot.py in the `lerobot` conda env (`--dataset {out}`).
  3. Write a preview GIF of episode 0 to {out}/preview.gif with scripts/preview_episodes.py.
  4. If `hf_repo` is non-empty ("{hf_repo}"), upload {out}/lerobot as a dataset repo with tags `{version}` and `v3.0`,
     as the guide describes (never upload provided Siemens data).
  5. Add one row to docs/decisions.md with: spec, split, window, expert success per object, episodes kept,
     throughput, output folder and HF link.
allowed_paths:
  - docs/decisions.md
  - docs/data.md
acceptance:
  - {name: dataset-exists, cmd: "test -f {out}/lerobot/meta/info.json"}
  - {name: manifest-exists, cmd: "ls {out}/raw/*/manifest.yaml >/dev/null || ls {out}/manifest.yaml >/dev/null"}
timeout_min: 120
```

- [ ] **Step 5: Create `harness/tasks/add_variation_axis.yaml`**

```yaml
id: add_variation_axis
title: Add a variation axis (domain randomization)
description: >
  Add a new per-episode randomization (e.g. camera jitter, unseen distractors, table texture) to EpisodeConfig,
  the variation spec and the robosuite randomizer, and expose it in the catalog.
catalog_target: variation
params:
  - {name: axis_id, type: str, help: "snake_case, e.g. camera_jitter"}
  - {name: label, type: str, help: "UI label, e.g. 'Camera jitter ±2 cm / ±3°'"}
  - {name: description, type: text, help: "what varies, ranges, train vs held-out values"}
  - {name: spec_out, type: str, default: cathlab_v2, help: "new or existing versioned spec (never edit cathlab_v1)"}
context:
  - AGENTS.md
  - docs/specs/2026-10-22-sim-framework-design.md
  - dataengine/variation.py
  - configs/variation/cathlab_v1.yaml
  - sim/robosuite/randomize.py
  - sim/robosuite/env.py
  - configs/catalog.yaml
prompt: |
  Add the variation axis `{axis_id}` ("{label}"): {description}
  1. Add its ranges/values to configs/variation/{spec_out}.yaml (create it as a copy of cathlab_v1.yaml if missing;
     never edit cathlab_v1.yaml) and the matching optional fields to VariationSpec / EpisodeConfig in
     dataengine/variation.py. Defaults must reproduce cathlab_v1 behaviour exactly (old specs keep working).
  2. Sample it in train_configs/eval_configs deterministically from the episode seed; keep held-out values out of training.
  3. Apply it in sim/robosuite/randomize.py (apply_episode_config) on every reset.
  4. In configs/catalog.yaml set the `{axis_id}` item in section `variation` to status ready (add the item if missing).
  5. Tests: unit test for the sampling in tests/dataengine/, and a robosuite test in tests/sim/ that two configs differ on the axis.
  6. One row in docs/decisions.md.
allowed_paths:
  - dataengine/variation.py
  - configs/variation/
  - configs/catalog.yaml
  - sim/robosuite/randomize.py
  - sim/robosuite/env.py
  - tests/
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
  - {name: robosuite-tests, cmd: "conda run -n robosuite --no-capture-output env MUJOCO_GL=egl python -m pytest -q tests/sim"}
  - name: expert-rate
    cmd: >-
      conda run -n robosuite --no-capture-output env MUJOCO_GL=egl python scripts/robosuite_expert_check.py
      --n 4 --spec configs/variation/{spec_out}.yaml --json var/expert_check.json
      && python3 scripts/assert_expert_rate.py var/expert_check.json --min 0.8
    timeout_s: 1800
timeout_min: 45
```

- [ ] **Step 6: Create `harness/tasks/add_sim_backend.yaml`**

```yaml
id: add_sim_backend
title: Add or extend a simulator backend
description: >
  Implement the expert adapter (dataengine.expert.ExpertSim) and the eval backend (contracts.SimBackend) for a simulator,
  following the robosuite reference implementation, and register it in the catalog.
catalog_target: simulator
params:
  - {name: sim, type: choice, choices: [isaac, mujoco], help: "package sim/<sim>/"}
  - {name: embodiment, type: str, default: fr3_duo}
  - {name: notes, type: text, required: false, help: "scene file, env name, known constraints"}
context:
  - AGENTS.md
  - docs/specs/2026-10-22-sim-framework-design.md
  - docs/handoffs/2026-10-22-isaac-fr3-session.md
  - contracts/sim.py
  - dataengine/expert.py
  - dataengine/variation.py
  - sim/robosuite/expert_sim.py
  - sim/robosuite/backend.py
  - sim/robosuite/README.md
prompt: |
  Build sim/{sim}/ for embodiment `{embodiment}`. Notes from the user: {notes}
  1. `sim/{sim}/expert_sim.py`: a class implementing the ExpertSim protocol (reset(cfg), get_object_pose, get_ee_pose,
     step, tray_contains), converting the simulator's frames to the core frame (robot base, table top z = 0, wxyz quats),
     exactly as sim/robosuite/expert_sim.py does. Import the simulator lazily so tests run without it.
  2. `sim/{sim}/backend.py`: contracts.SimBackend for policy eval with stage flags reach/grasp/lift/transport/place.
  3. `sim/{sim}/README.md` (env setup, how to run, conventions) and tests in tests/sim/ that skip without the simulator.
  4. Update the `{sim}` item in configs/catalog.yaml: status in_progress until the expert reaches 80 %, then ready
     (state the measured rate in docs/decisions.md).
allowed_paths:
  - sim/{sim}/
  - tests/sim/
  - configs/
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 60
```

- [ ] **Step 7: Create `harness/tasks/add_policy_adapter.yaml`**

```yaml
id: add_policy_adapter
title: Add a policy model adapter
description: >
  Wrap a model so the pipeline can train it and serve it through the policy server: a contracts.PolicyAdapter plus
  the training command, registered in the catalog.
catalog_target: model
params:
  - {name: model_id, type: str, help: "catalog id, e.g. pi05"}
  - {name: lerobot_policy_type, type: str, required: false, help: "e.g. pi05 / diffusion; empty if not LeRobot-native"}
  - {name: notes, type: text, required: false}
context:
  - AGENTS.md
  - contracts/policy.py
  - docs/research/vla_selection.md
  - docs/research/adapter_notes_for_rakesh.md
  - docs/specs/2026-10-23-control-station-design.md
  - configs/catalog.yaml
prompt: |
  Add model `{model_id}` (LeRobot policy type: "{lerobot_policy_type}"). Notes: {notes}
  1. `train/adapters/{model_id}.py`: a contracts.PolicyAdapter that loads a checkpoint and returns an action chunk for
     the robosuite Panda I/O (8-D state, 7-D normalized EEF-delta action, agentview + eye_in_hand 256x256). Import the
     model library lazily.
  2. `train/recipes/{model_id}.yaml`: the training command/arguments (lerobot-train flags) with sensible defaults for a
     40 GB GPU and a small smoke setting.
  3. A GPU-free test in tests/train/ that checks the adapter's meta() and argument building without loading weights.
  4. Set the catalog item `{model_id}` in section `model` to ready (add it if missing).
allowed_paths:
  - train/
  - tests/train/
  - configs/catalog.yaml
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 45
```

- [ ] **Step 8: Create `harness/tasks/add_expert.yaml`**

```yaml
id: add_expert
title: Add an expert controller variant (e.g. GraspGen)
description: >
  Add an alternative grasp source for the scripted expert (e.g. GraspGen candidate ranking), selectable by name,
  without changing the default analytic expert's behaviour.
catalog_target: expert
params:
  - {name: expert_id, type: str, default: graspgen}
  - {name: notes, type: text, required: false, help: "install notes, model weights location, GPU needs"}
context:
  - AGENTS.md
  - dataengine/expert.py
  - dataengine/variation.py
  - sim/robosuite/expert_sim.py
  - sim/robosuite/params.py
  - scripts/robosuite_expert_check.py
  - configs/catalog.yaml
prompt: |
  Add the expert variant `{expert_id}`. Notes: {notes}
  1. `dataengine/experts/{expert_id}.py`: a grasp provider returning (tcp_position, tcp_quat, approach) like
     dataengine.expert.grasp_pose, using the simulator's object pose/point cloud. Heavy dependencies are imported lazily.
  2. Let run_expert_episode / plan_waypoints accept an optional grasp provider; default stays the analytic templates
     (all existing tests must pass unchanged).
  3. `--expert {expert_id}` option in scripts/robosuite_expert_check.py and scripts/robosuite_generate.py.
  4. Tests in tests/dataengine/ with a fake provider. Catalog item `{expert_id}` in section `expert`: in_progress until
     the robosuite expert check reaches 80 % with it, then ready. One row in docs/decisions.md with the measured rate.
allowed_paths:
  - dataengine/
  - scripts/robosuite_expert_check.py
  - scripts/robosuite_generate.py
  - tests/
  - configs/catalog.yaml
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
  - {name: robosuite-tests, cmd: "conda run -n robosuite --no-capture-output env MUJOCO_GL=egl python -m pytest -q tests/sim"}
timeout_min: 60
```

- [ ] **Step 9: Create `harness/tasks/free_task.yaml`**

```yaml
id: free_task
title: Free task (your own prompt and paths)
description: >
  Any development task in this repo. You write the goal and list the paths the agent may change; the default
  acceptance check is the GPU-free core test suite.
catalog_target: ""
params:
  - {name: goal, type: text, help: "what to build or fix, with done-criteria"}
  - {name: paths, type: str, help: "comma-separated allowed paths, e.g. 'eval/, tests/eval/, docs/decisions.md'"}
  - {name: check, type: str, default: "uv run pytest -q", help: "acceptance command"}
context:
  - AGENTS.md
  - docs/specs/2026-10-23-control-station-design.md
prompt: |
  {goal}
allowed_paths:
  - "{paths}"
acceptance:
  - {name: check, cmd: "{check}"}
timeout_min: 45
```

- [ ] **Step 10: Create `scripts/assert_expert_rate.py`**

```python
"""Exit 1 unless every object in a robosuite_expert_check.py --json summary reaches the minimum success rate."""
import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    ap.add_argument("--min", type=float, default=0.8)
    args = ap.parse_args(argv)
    per_object = json.loads(Path(args.summary).read_text())["per_object"]
    low = {o: v["rate"] for o, v in per_object.items() if v["rate"] < args.min}
    for o, v in per_object.items():
        print(f"{o:12s} {v['rate']:.0%} ({v['success']}/{v['attempted']})")
    if low:
        print(f"FAIL: below {args.min:.0%}: {low}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 11: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/harness/test_templates.py`
Expected: 11 passed

- [ ] **Step 12: Commit**

```bash
git add harness/tasks scripts/assert_expert_rate.py tests/harness/test_templates.py
git commit -m "feat(harness): seven starter task templates and expert-rate check"
```

### Task 10: Industrial UI shell (PROJECT, PIPELINE strip, DATA, MODELS, AGENT, LOG) and safe demo

**Files:**
- Create: `frontend/index.html`, `frontend/css/station.css`, `frontend/js/api.js`, `frontend/js/dom.js`, `frontend/js/app.js`, `frontend/js/agent.js`, `scripts/demo_control_station.py`

**Interfaces:**
- Consumes: Task 8 `/api/catalog`, `/api/selection/check`, `/api/runs`, `/api/jobs`, `/api/workers`; Task 7 `/api/agent/*`.
- Produces: the UI at `/`. `dom.js` exports `$`, `el`, `setStatus`. `agent.js` exports `initAgent()` and `openTemplate(taskId, prefill)`. **`agent.js` and `app.js` must not import each other's helpers; both import `dom.js`, because a cycle breaks top-level code in `agent.js`.**

- [ ] **Step 1: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CathLab Control Station</title>
  <link rel="stylesheet" href="css/station.css">
</head>
<body>
  <header class="bar top">
    <div class="brand">CATHLAB&nbsp;CS</div>
    <nav class="tabs" id="tabs">
      <button data-tab="project" class="active">PROJECT</button>
      <button data-tab="pipeline">PIPELINE</button>
      <button data-tab="data">DATA</button>
      <button data-tab="models">MODELS</button>
      <button data-tab="agent">AGENT</button>
      <button data-tab="log">LOG</button>
    </nav>
    <div class="workers" id="workers" title="Worker health"></div>
  </header>

  <main>
    <!-- PROJECT: catalog tree + section editor -->
    <section class="tab" id="tab-project">
      <aside class="tree">
        <div class="panel-title">PROJECT TREE</div>
        <ul id="tree"></ul>
        <div class="panel-title">CURRENT RUN</div>
        <div class="kv mono" id="current-run">none</div>
      </aside>
      <div class="pane">
        <div class="panel-title" id="section-title">SECTION</div>
        <div id="section-items" class="items"></div>
        <div id="section-add"></div>
        <div class="panel-title">SELECTION CHECK</div>
        <div id="problems" class="problems"></div>
        <div class="row">
          <input id="run-note" class="field" placeholder="Run note (optional)">
          <button class="btn primary" id="btn-create-run">CREATE RUN</button>
        </div>
      </div>
    </section>

    <!-- PIPELINE: stage strip -->
    <section class="tab hidden" id="tab-pipeline">
      <div class="strip" id="strip"></div>
      <div class="pane">
        <div class="panel-title" id="stage-title">STAGE</div>
        <div id="stage-detail" class="muted">Select a stage.</div>
      </div>
    </section>

    <section class="tab hidden" id="tab-data">
      <div class="pane">
        <div class="panel-title">DATASETS</div>
        <div class="muted">No datasets registered yet. Generate and Intel stages populate this view.</div>
      </div>
    </section>

    <section class="tab hidden" id="tab-models">
      <div class="pane">
        <div class="panel-title">MODEL REGISTRY</div>
        <div class="muted">No model versions registered yet. Train, Eval and Promote populate this view.</div>
      </div>
    </section>

    <!-- AGENT: harness -->
    <section class="tab hidden" id="tab-agent">
      <aside class="tree">
        <div class="panel-title">TASK TEMPLATES</div>
        <ul id="templates"></ul>
        <div class="panel-title">AGENT JOBS</div>
        <ul id="agent-jobs"></ul>
      </aside>
      <div class="pane">
        <div id="agent-form-wrap">
          <div class="panel-title" id="agent-form-title">NEW AGENT TASK</div>
          <div class="muted" id="agent-form-desc">Pick a template on the left.</div>
          <form id="agent-form" class="form"></form>
        </div>
        <div id="agent-job-wrap" class="hidden">
          <div class="panel-title" id="agent-job-title">AGENT JOB</div>
          <div class="kv mono" id="agent-job-meta"></div>
          <div class="row" id="agent-job-actions">
            <button class="btn" id="btn-view-prompt">PROMPT</button>
            <button class="btn" id="btn-view-diff">DIFF</button>
            <button class="btn ok" id="btn-push">PUSH BRANCH</button>
            <button class="btn danger" id="btn-discard">DISCARD</button>
          </div>
          <pre class="console" id="agent-console"></pre>
        </div>
      </div>
    </section>

    <!-- LOG -->
    <section class="tab hidden" id="tab-log">
      <div class="pane">
        <div class="panel-title">JOBS</div>
        <table class="grid" id="jobs-table">
          <thead><tr><th>ID</th><th>KIND</th><th>NAME</th><th>STATE</th><th>UPDATED</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </section>
  </main>

  <footer class="bar bottom">
    <button class="btn run" id="btn-run" disabled title="Pipeline stages arrive with Plan B">&#9654; RUN PIPELINE</button>
    <button class="btn stop" id="btn-stop" disabled>&#9632; STOP</button>
    <div class="status mono" id="status">READY</div>
  </footer>

  <script type="module" src="js/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `frontend/css/station.css`**

```css
/* CathLab Control Station: industrial panel style (flat, high contrast, no gradients). */
:root {
  --bg: #e6e8eb; --panel: #f4f5f7; --panel-2: #d5d9de; --line: #9aa1a9; --ink: #1d2733;
  --navy: #2b3a4a; --blue: #0d6efd; --ok: #2e7d32; --warn: #b26a00; --bad: #c62828; --muted: #6b7380;
  --mono: "IBM Plex Mono", "DejaVu Sans Mono", Consolas, monospace;
  --sans: "IBM Plex Sans", "Segoe UI", Arial, Helvetica, sans-serif;
}
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; }
body { display: flex; flex-direction: column; background: var(--bg); color: var(--ink); font: 14px/1.4 var(--sans); }
.mono { font-family: var(--mono); }
.hidden { display: none !important; }
.muted { color: var(--muted); }

.bar { display: flex; align-items: stretch; background: var(--navy); color: #fff; min-height: 44px; }
.bar.bottom { align-items: center; gap: 8px; padding: 6px 10px; }
.brand { padding: 12px 16px; font-weight: 700; letter-spacing: 1px; border-right: 1px solid #3d4f62; }
.tabs { display: flex; }
.tabs button { background: transparent; color: #fff; border: 0; border-right: 1px solid #3d4f62; padding: 0 18px;
  font: 600 13px var(--sans); letter-spacing: .6px; cursor: pointer; }
.tabs button.active { background: var(--blue); }
.tabs button:hover:not(.active) { background: #3a4d61; }
.workers { margin-left: auto; display: flex; align-items: center; gap: 12px; padding: 0 14px; font: 12px var(--mono); }
.lamp { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; border: 1px solid #0003; }
.lamp.ok { background: #2ecc71; } .lamp.off { background: #7d8792; } .lamp.warn { background: #f0a020; } .lamp.bad { background: #e5484d; }

main { flex: 1; overflow: hidden; }
.tab { display: flex; height: 100%; }
#tab-pipeline { flex-direction: column; }
.tree { width: 300px; background: var(--panel-2); border-right: 1px solid var(--line); overflow: auto; padding: 10px; }
.tree ul { list-style: none; margin: 0 0 14px; padding: 0; }
.tree li { padding: 7px 8px; border-left: 3px solid transparent; cursor: pointer; display: flex; justify-content: space-between; gap: 8px; }
.tree li:hover { background: #e9ecef; }
.tree li.active { background: #fff; border-left-color: var(--blue); }
.tree li .val { font: 12px var(--mono); color: var(--muted); text-align: right; }
.pane { flex: 1; overflow: auto; padding: 14px 18px; background: var(--panel); }
.panel-title { font-weight: 700; color: var(--navy); letter-spacing: .6px; font-size: 12px; text-transform: uppercase;
  border-bottom: 1px solid var(--line); padding-bottom: 4px; margin: 6px 0 10px; }
.kv { background: #fff; border: 1px solid var(--line); padding: 8px; margin-bottom: 12px; font-size: 12px; white-space: pre-wrap; }

.items { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 8px; margin-bottom: 12px; }
.item { background: #fff; border: 1px solid var(--line); padding: 10px; cursor: pointer; min-height: 74px; }
.item.selected { border: 2px solid var(--blue); padding: 9px; }
.item.disabled { background: #eceef1; color: var(--muted); }
.item .label { font-weight: 600; }
.item .notes { font-size: 12px; color: var(--muted); }
.item .state { font: 11px var(--mono); margin-top: 6px; }
.state.ready { color: var(--ok); } .state.in_progress { color: var(--warn); } .state.planned { color: var(--muted); }

.btn { background: #fff; color: var(--ink); border: 1px solid var(--line); padding: 8px 14px; font: 600 12px var(--sans);
  letter-spacing: .5px; cursor: pointer; text-transform: uppercase; }
.btn:hover:not(:disabled) { border-color: var(--navy); }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn.primary { background: var(--blue); color: #fff; border-color: var(--blue); }
.btn.ok, .btn.run { background: var(--ok); color: #fff; border-color: var(--ok); }
.btn.danger, .btn.stop { background: var(--bad); color: #fff; border-color: var(--bad); }
.btn.agent { background: #fff; color: var(--blue); border: 1px dashed var(--blue); }
.row { display: flex; gap: 8px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
.field { flex: 1; min-width: 220px; padding: 8px; border: 1px solid var(--line); font: 13px var(--sans); background: #fff; }
.status { margin-left: auto; }

.problems .p { border-left: 4px solid var(--warn); background: #fff; padding: 6px 8px; margin-bottom: 6px; display: flex;
  justify-content: space-between; align-items: center; gap: 8px; }
.problems .ok { border-left: 4px solid var(--ok); background: #fff; padding: 6px 8px; }

.form { display: grid; grid-template-columns: 220px 1fr; gap: 8px 12px; align-items: start; max-width: 900px; }
.form label { font-weight: 600; padding-top: 7px; }
.form .help { grid-column: 2; font-size: 12px; color: var(--muted); margin-top: -4px; }
.form textarea { min-height: 80px; }
.form .actions { grid-column: 1 / span 2; }

.console { background: #11161c; color: #d8dee6; font: 12px/1.45 var(--mono); padding: 10px; min-height: 260px;
  max-height: 55vh; overflow: auto; white-space: pre-wrap; border: 1px solid #000; }

.strip { display: flex; gap: 4px; padding: 8px; background: var(--panel-2); border-bottom: 1px solid var(--line); }
.stage { flex: 1; background: #fff; border-top: 5px solid var(--line); padding: 8px; text-align: center; cursor: pointer; font-weight: 600; }
.stage .sub { font: 11px var(--mono); color: var(--muted); }
.stage.done { border-top-color: var(--ok); } .stage.running { border-top-color: var(--blue); }
.stage.failed { border-top-color: var(--bad); } .stage.blocked { background: #eceef1; color: var(--muted); }
.stage.active { outline: 2px solid var(--blue); }

table.grid { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
table.grid th, table.grid td { border: 1px solid var(--line); padding: 6px 8px; text-align: left; }
table.grid th { background: var(--panel-2); font-size: 12px; letter-spacing: .5px; }
.badge { font: 11px var(--mono); padding: 2px 6px; border: 1px solid var(--line); background: #fff; }
.badge.ready_for_review, .badge.done, .badge.pushed { border-color: var(--ok); color: var(--ok); }
.badge.failed, .badge.cancelled, .badge.timeout { border-color: var(--bad); color: var(--bad); }
.badge.agent_running, .badge.checks, .badge.running, .badge.worktree, .badge.guard { border-color: var(--blue); color: var(--blue); }
```

- [ ] **Step 3: Create `frontend/js/api.js`**

```javascript
// Minimal JSON API client for the control plane.
async function request(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  const type = res.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = data && data.detail !== undefined ? data.detail : data;
    const err = new Error(typeof detail === "string" ? detail : (detail.message || JSON.stringify(detail)));
    err.detail = detail;
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  get: (p) => request("GET", p),
  post: (p, b) => request("POST", p, b === undefined ? {} : b),
};
```

- [ ] **Step 4: Create `frontend/js/dom.js`**

```javascript
// DOM helpers shared by all views.
export const $ = (s) => document.querySelector(s);

export function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function setStatus(text) {
  $("#status").textContent = text;
}
```

- [ ] **Step 5: Create `frontend/js/app.js`**

```javascript
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { initAgent, openTemplate } from "./agent.js";

// ---- tabs -------------------------------------------------------------------------------
export function showTab(name) {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== `tab-${name}`));
}
document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---- PROJECT tab --------------------------------------------------------------------------
const state = { catalog: null, selection: {}, section: null, run: null };

function valueText(section, value) {
  if (Array.isArray(value)) return value.length ? `${value.length} selected` : "none";
  const item = section.items.find((i) => i.id === value);
  return item ? item.label : String(value);
}

function renderTree() {
  const ul = $("#tree");
  ul.innerHTML = "";
  for (const [key, section] of Object.entries(state.catalog.sections)) {
    const li = el("li", key === state.section ? "active" : "");
    li.append(el("span", "", section.title), el("span", "val", valueText(section, state.selection[key])));
    li.addEventListener("click", () => { state.section = key; renderTree(); renderSection(); });
    ul.append(li);
  }
}

function toggleItem(key, section, id) {
  if (section.multi) {
    const cur = new Set(state.selection[key] || []);
    cur.has(id) ? cur.delete(id) : cur.add(id);
    state.selection[key] = section.items.map((i) => i.id).filter((x) => cur.has(x));
  } else {
    state.selection[key] = id;
  }
  renderTree();
  renderSection();
  checkSelection();
}

function renderSection() {
  const key = state.section;
  const section = state.catalog.sections[key];
  $("#section-title").textContent = section.title + (section.multi ? " (multi-select)" : "");
  const box = $("#section-items");
  box.innerHTML = "";
  const chosen = section.multi ? new Set(state.selection[key] || []) : new Set([state.selection[key]]);
  for (const item of section.items) {
    const card = el("div", "item" + (chosen.has(item.id) ? " selected" : "") + (item.status === "planned" ? " disabled" : ""));
    card.append(el("div", "label", item.label));
    if (item.notes) card.append(el("div", "notes", item.notes));
    const st = el("div", `state ${item.status}`, "● " + item.status.replace("_", " ").toUpperCase());
    card.append(st);
    if (item.status !== "ready" && item.harness_task) {
      const b = el("button", "btn agent", "Build with agent");
      b.addEventListener("click", (ev) => { ev.stopPropagation(); buildWithAgent(item.harness_task, key, item.id); });
      card.append(b);
    }
    card.addEventListener("click", () => toggleItem(key, section, item.id));
    box.append(card);
  }
  const add = $("#section-add");
  add.innerHTML = "";
  if (section.add_task) {
    const b = el("button", "btn agent", `+ Add ${section.title.toLowerCase()} with agent`);
    b.addEventListener("click", () => buildWithAgent(section.add_task, key, null));
    add.append(b);
  }
}

function buildWithAgent(taskId, sectionKey, itemId) {
  const prefill = {};
  if (taskId === "add_sim_backend" && itemId) prefill.sim = itemId;
  if (taskId === "add_policy_adapter" && itemId) prefill.model_id = itemId;
  if (taskId === "add_expert" && itemId) prefill.expert_id = itemId;
  if (taskId === "add_variation_axis" && itemId) prefill.axis_id = itemId;
  showTab("agent");
  openTemplate(taskId, prefill);
}

async function checkSelection() {
  const { problems } = await api.post("/api/selection/check", { selection: state.selection });
  const box = $("#problems");
  box.innerHTML = "";
  if (!problems.length) {
    box.append(el("div", "ok", "✓ Selection is runnable."));
  }
  for (const p of problems) {
    const row = el("div", "p");
    row.append(el("span", "", p.message));
    if (p.harness_task) {
      const b = el("button", "btn agent", "Build with agent");
      b.addEventListener("click", () => buildWithAgent(p.harness_task, p.section, p.item));
      row.append(b);
    }
    box.append(row);
  }
  $("#btn-create-run").disabled = problems.length > 0;
}

async function createRun() {
  try {
    const run = await api.post("/api/runs", { selection: state.selection, note: $("#run-note").value });
    state.run = run;
    $("#current-run").textContent = `${run.id}\n${Object.entries(run.selection)
      .map(([k, v]) => `${k}: ${Array.isArray(v) ? v.join(", ") : v}`).join("\n")}`;
    setStatus(`RUN ${run.id} CREATED`);
  } catch (e) {
    setStatus(`RUN REJECTED: ${e.message}`);
  }
}

// ---- PIPELINE tab (stage strip; stages are wired in Plan B) -----------------------------------
const STAGES = ["SCENE", "GENERATE", "INTEL", "TRAIN", "EVAL", "PROMOTE", "ROLLOUT"];
function renderStrip() {
  const strip = $("#strip");
  strip.innerHTML = "";
  STAGES.forEach((s) => {
    const t = el("div", "stage blocked");
    t.append(el("div", "", s), el("div", "sub", "not wired"));
    t.addEventListener("click", () => {
      document.querySelectorAll(".stage").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      $("#stage-title").textContent = s;
      $("#stage-detail").textContent = "Stage execution arrives with Plan B (stages, eval, registry, rollout).";
    });
    strip.append(t);
  });
}

// ---- workers + LOG --------------------------------------------------------------------------
async function refreshWorkers() {
  try {
    const ws = await api.get("/api/workers");
    const box = $("#workers");
    box.innerHTML = "";
    for (const w of ws) {
      const span = el("span", "");
      span.title = w.healthy ? `${w.url}\n${(w.capabilities || []).join(", ")}` : `${w.url}\n${w.error}`;
      span.append(el("span", `lamp ${w.healthy ? "ok" : "off"}`), document.createTextNode(w.name.toUpperCase()));
      box.append(span);
    }
  } catch (e) { /* control plane restarting */ }
}

async function refreshJobs() {
  const [jobs, agentJobs] = await Promise.all([
    api.get("/api/jobs"),
    api.get("/api/agent/jobs").catch(() => []),
  ]);
  const rows = [
    ...jobs.map((j) => ({ id: j.id, kind: j.kind, name: j.name, state: j.state, updated: j.updated })),
    ...agentJobs.map((j) => ({ id: j.id, kind: "agent", name: j.task, state: j.state, updated: j.updated })),
  ].sort((a, b) => b.updated - a.updated);
  const tbody = $("#jobs-table tbody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = el("tr");
    const st = el("td");
    st.append(el("span", `badge ${r.state}`, r.state));
    tr.append(el("td", "mono", r.id), el("td", "", r.kind), el("td", "", r.name), st,
      el("td", "mono", new Date(r.updated * 1000).toLocaleTimeString()));
    tbody.append(tr);
  }
}

// ---- boot -----------------------------------------------------------------------------------
async function boot() {
  state.catalog = await api.get("/api/catalog");
  state.selection = JSON.parse(JSON.stringify(state.catalog.defaults));
  state.section = Object.keys(state.catalog.sections)[0];
  renderTree();
  renderSection();
  renderStrip();
  await checkSelection();
  $("#btn-create-run").addEventListener("click", createRun);
  await initAgent();
  refreshWorkers();
  refreshJobs();
  setInterval(refreshWorkers, 5000);
  setInterval(refreshJobs, 4000);
}
boot().catch((e) => setStatus(`ERROR: ${e.message}`));
```

- [ ] **Step 6: Create `frontend/js/agent.js`**

```javascript
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";

const st = { templates: {}, agents: null, current: null, logOffset: 0, timer: null };
const ACTIVE = new Set(["created", "worktree", "agent_running", "guard", "checks"]);

export async function initAgent() {
  try {
    const [tasks, agents] = await Promise.all([api.get("/api/agent/tasks"), api.get("/api/agent/agents")]);
    tasks.forEach((t) => { st.templates[t.id] = t; });
    st.agents = agents;
  } catch (e) {
    $("#agent-form-desc").textContent = "Agent harness is not mounted on this control plane (start it with the agent enabled).";
    return;
  }
  renderTemplates();
  refreshAgentJobs();
  setInterval(refreshAgentJobs, 3000);
}

function renderTemplates() {
  const ul = $("#templates");
  ul.innerHTML = "";
  for (const t of Object.values(st.templates)) {
    const li = el("li", "");
    li.append(el("span", "", t.title));
    li.addEventListener("click", () => openTemplate(t.id, {}));
    ul.append(li);
  }
}

function inputFor(p, value) {
  let input;
  if (p.type === "choice") {
    input = el("select", "field");
    p.choices.forEach((c) => { const o = el("option", "", c); o.value = c; input.append(o); });
  } else if (p.type === "text") {
    input = el("textarea", "field");
  } else if (p.type === "bool") {
    input = el("select", "field");
    ["true", "false"].forEach((c) => { const o = el("option", "", c); o.value = c; input.append(o); });
  } else {
    input = el("input", "field");
    if (p.type === "int" || p.type === "float") input.type = "number";
    if (p.type === "float") input.step = "any";
  }
  input.name = p.name;
  const v = value !== undefined ? value : p.default;
  if (v !== null && v !== undefined) input.value = String(v);
  return input;
}

export function openTemplate(taskId, prefill) {
  const t = st.templates[taskId];
  $("#agent-job-wrap").classList.add("hidden");
  $("#agent-form-wrap").classList.remove("hidden");
  if (!t) {
    $("#agent-form-title").textContent = "UNKNOWN TASK";
    $("#agent-form-desc").textContent = `Template '${taskId}' not found.`;
    $("#agent-form").innerHTML = "";
    return;
  }
  $("#agent-form-title").textContent = t.title;
  $("#agent-form-desc").textContent = t.description;
  const form = $("#agent-form");
  form.innerHTML = "";
  for (const p of t.params) {
    form.append(el("label", "", p.name + (p.required ? " *" : "")), inputFor(p, prefill[p.name]));
    if (p.help) form.append(el("div", "help", p.help));
  }
  form.append(el("label", "", "extra instructions"));
  const extra = el("textarea", "field");
  extra.name = "__extra";
  form.append(extra);
  form.append(el("label", "", "agent"));
  const sel = el("select", "field");
  sel.name = "__agent";
  (st.agents?.agents || []).forEach((a) => {
    const o = el("option", "", `${a.name}: ${a.description}`);
    o.value = a.name;
    if (a.name === st.agents.default) o.selected = true;
    sel.append(o);
  });
  form.append(sel);
  const paths = el("div", "help", "Allowed paths: " + t.allowed_paths.join(", ") +
    " · Checks: " + t.acceptance.map((c) => c.name).join(", "));
  form.append(el("div", ""), paths);
  const actions = el("div", "actions row");
  const go = el("button", "btn primary", "Start agent");
  go.type = "submit";
  actions.append(go);
  form.append(actions);
  form.onsubmit = async (ev) => {
    ev.preventDefault();
    const fd = new FormData(form);
    const params = {};
    for (const p of t.params) {
      const v = fd.get(p.name);
      if (v !== null && v !== "") params[p.name] = v;
    }
    try {
      const job = await api.post("/api/agent/jobs", { task: t.id, params, extra: fd.get("__extra") || "", agent: fd.get("__agent") });
      setStatus(`AGENT JOB ${job.id} STARTED`);
      openJob(job.id);
      refreshAgentJobs();
    } catch (e) {
      setStatus(`AGENT REJECTED: ${e.message}`);
    }
  };
}

async function refreshAgentJobs() {
  const jobs = await api.get("/api/agent/jobs").catch(() => []);
  const ul = $("#agent-jobs");
  ul.innerHTML = "";
  for (const j of jobs) {
    const li = el("li", j.id === st.current ? "active" : "");
    li.append(el("span", "mono", `${j.task}`), el("span", `badge ${j.state}`, j.state));
    li.addEventListener("click", () => openJob(j.id));
    ul.append(li);
  }
}

async function openJob(id) {
  st.current = id;
  st.logOffset = 0;
  $("#agent-console").textContent = "";
  $("#agent-form-wrap").classList.add("hidden");
  $("#agent-job-wrap").classList.remove("hidden");
  if (st.timer) clearInterval(st.timer);
  await pollJob();
  st.timer = setInterval(pollJob, 1500);
}

async function pollJob() {
  const id = st.current;
  if (!id) return;
  const job = await api.get(`/api/agent/jobs/${id}`);
  $("#agent-job-title").textContent = `${job.title} · ${job.id}`;
  const checks = job.checks.map((c) => `${c.ok ? "✓" : "✕"} ${c.name}`).join("  ") || "-";
  $("#agent-job-meta").textContent =
    `STATE    ${job.state}${job.error ? "\nERROR    " + job.error : ""}\nAGENT    ${job.agent}\nBRANCH   ${job.branch}` +
    `\nFILES    ${job.changed_files.join(", ") || "-"}\nCHECKS   ${checks}`;
  $("#btn-push").disabled = job.state !== "ready_for_review";
  $("#btn-discard").disabled = ACTIVE.has(job.state) || job.state === "discarded" || job.state === "pushed";
  const log = await api.get(`/api/agent/jobs/${id}/log?offset=${st.logOffset}`);
  if (log.text) {
    const c = $("#agent-console");
    c.textContent += log.text;
    c.scrollTop = c.scrollHeight;
  }
  st.logOffset = log.offset;
  if (!ACTIVE.has(job.state) && st.timer) { clearInterval(st.timer); st.timer = null; }
}

async function showText(path) {
  const text = await api.get(path);
  $("#agent-console").textContent = text || "(empty)";
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
}

$("#btn-view-prompt").addEventListener("click", () => st.current && showText(`/api/agent/jobs/${st.current}/prompt`));
$("#btn-view-diff").addEventListener("click", () => st.current && showText(`/api/agent/jobs/${st.current}/diff`));
$("#btn-push").addEventListener("click", async () => {
  try { const j = await api.post(`/api/agent/jobs/${st.current}/push`); setStatus(`PUSHED ${j.branch}`); pollJob(); }
  catch (e) { setStatus(`PUSH FAILED: ${e.message}`); }
});
$("#btn-discard").addEventListener("click", async () => {
  try { await api.post(`/api/agent/jobs/${st.current}/discard`); setStatus("DISCARDED"); pollJob(); refreshAgentJobs(); }
  catch (e) { setStatus(`DISCARD FAILED: ${e.message}`); }
});
```

- [ ] **Step 7: Create `scripts/demo_control_station.py`**

```python
"""Try the Control Station UI safely: a throwaway git repo and a fake agent (no Copilot login, no real repo changes).

    uv run python scripts/demo_control_station.py      ->  http://localhost:8700

Pick AGENT -> "Free task", set paths `configs/` and check `test -f configs/new.txt`, then Start agent.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from harness.agents import AgentConfig, AgentsConfig  # noqa: E402
from harness.model import load_templates  # noqa: E402
from harness.runner import Harness, HarnessSettings  # noqa: E402
from orchestrator.control import create_control_app  # noqa: E402

FAKE_AGENT = (
    "import sys, pathlib, time\n"
    "print('fake agent working...'); time.sleep(2)\n"
    "p = pathlib.Path(sys.argv[1]) / 'configs/new.txt'\n"
    "p.parent.mkdir(parents=True, exist_ok=True)\n"
    "p.write_text('hello from fake agent\\n'); print('done')\n"
)


def sh(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cathlab-demo-"))
    sh("git", "init", "-q", "--bare", "-b", "main", str(tmp / "origin.git"), cwd=tmp)
    repo = tmp / "repo"
    sh("git", "init", "-q", "-b", "main", str(repo), cwd=tmp)
    sh("git", "config", "user.name", "Demo", cwd=repo)
    sh("git", "config", "user.email", "demo@example.com", cwd=repo)
    (repo / "README.md").write_text("demo\n")
    sh("git", "add", "-A", cwd=repo)
    sh("git", "commit", "-q", "-m", "init", cwd=repo)
    sh("git", "remote", "add", "origin", str(tmp / "origin.git"), cwd=repo)
    sh("git", "push", "-q", "-u", "origin", "main", cwd=repo)
    agent_py = tmp / "fake_agent.py"
    agent_py.write_text(FAKE_AGENT)
    agents = AgentsConfig(default="fake", agents={"fake": AgentConfig(
        name="fake", description="demo agent (writes configs/new.txt)", argv=[sys.executable, str(agent_py), "{workdir}"])})
    harness = Harness(HarnessSettings(repo=repo, var_dir=tmp / "var", worktrees_dir=tmp / "wts"),
                      load_templates(ROOT / "harness/tasks"), agents)
    print(f"demo repo: {repo}\nopen http://localhost:8700")
    uvicorn.run(create_control_app(ROOT, var_dir=tmp / "control", harness=harness), host="127.0.0.1", port=8700,
                log_level="warning")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Syntax-check the JavaScript**

```bash
for f in frontend/js/*.js; do node --input-type=module --check < "$f" || exit 1; done && echo "js ok"
```

If `node` is missing: `conda create -y -n node nodejs=20` and use `conda run -n node node …`.
Expected: `js ok`.

- [ ] **Step 9: Verify in a browser with the demo (no real agent, no repo changes)**

Run `uv run python scripts/demo_control_station.py` and open http://localhost:8700. Check:
1. **PROJECT** shows the tree (6 sections) and the robosuite / Isaac / MuJoCo cards. Isaac and MuJoCo show **BUILD WITH AGENT**, and "✓ Selection is runnable" appears.
2. Clicking the Isaac card lists "Isaac Sim is in progress" (with Build with agent) and the Panda `requires` problem, and **CREATE RUN** is disabled.
3. **Build with agent** on the Isaac problem opens AGENT with "Add or extend a simulator backend" and `sim = isaac` prefilled.
4. AGENT → **Free task**:
   - goal "Create configs/new.txt", paths `configs/`, check `test -f configs/new.txt`, then **Start agent**;
   - the log streams, and the state goes to `ready_for_review` with `✓ check`;
   - **DIFF** shows `+hello from fake agent`;
   - **PUSH BRANCH** sets the state to `pushed` and disables PUSH and DISCARD.
5. The header worker lamps are grey (no workers running), with tooltips showing the connection error.

Stop the demo (Ctrl-C).

- [ ] **Step 10: Commit**

```bash
git add frontend scripts/demo_control_station.py
git commit -m "feat(frontend): industrial Control Station UI shell with agent console and safe demo"
```

---

### Task 11: Dev commands, install and team docs, credits; publish to `main`

**Files:**
- Create: `Makefile`, `docs/INSTALL.md`, `docs/HARNESS.md`, `CREDITS.md`
- Modify: `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: every earlier task.
- Produces: `make install | test | dev | worker-robosuite | worker-lerobot | agent | demo-ui | harness-list`.

- [ ] **Step 1: Create `Makefile`**

```makefile
# CathLab Control Station: common commands. Run from the repo root.
# Native (no Docker) dev setup; see docs/INSTALL.md.

PORT ?= 8700
ROBOSUITE_ENV ?= robosuite
LEROBOT_ENV ?= lerobot

.PHONY: install test dev worker-robosuite worker-lerobot agent demo-ui harness-list

install:            ## core env (uv) with dev deps
	uv sync

test:               ## GPU-free test suite
	uv run pytest -q

dev:                ## control plane + UI + agent harness on http://localhost:$(PORT)
	uv run python -m orchestrator.control --port $(PORT)

worker-robosuite:   ## robosuite worker (scene, generate, eval-rollout) on :8711
	MUJOCO_GL=egl conda run -n $(ROBOSUITE_ENV) --no-capture-output python -m orchestrator.worker \
		--name robosuite --capabilities scene,generate,eval --port 8711

worker-lerobot:     ## lerobot worker (convert, intel, train, policy-server) on :8712
	conda run -n $(LEROBOT_ENV) --no-capture-output python -m orchestrator.worker \
		--name lerobot --capabilities convert,intel,train,policy_server --port 8712

agent:              ## standalone agent worker (harness API only) on :8790
	uv run python -m harness serve --port 8790

demo-ui:            ## UI with a throwaway repo and a fake agent (no Copilot login needed)
	uv run python scripts/demo_control_station.py

harness-list:       ## list agent task templates
	uv run python -m harness list
```

- [ ] **Step 2: Create `docs/INSTALL.md`**

````markdown
# Installation

Two ways to run the CathLab Control Station:

| | Native (today) | Docker |
|---|---|---|
| Control plane + UI + agent harness | `uv` env on the host | `control` + `frontend` containers; the agent harness stays on the host |
| robosuite worker | conda env `robosuite` | `worker-robosuite` image |
| lerobot worker (convert, train, policy server) | conda env `lerobot` (laptop or GPU box) | `worker-lerobot` image (GPU) |

The Docker images and `compose.yaml` arrive with the Docker task; this page then gets a Docker section.

## 1. Prerequisites

- Linux, Python 3.10+, git, [uv](https://docs.astral.sh/uv/), conda (Miniforge recommended).
- For the agent harness: an agent CLI logged in on the host. Default: [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/use-copilot-agents/use-copilot-cli) (`copilot` then `/login`). Optional: Claude Code or Codex (see `configs/agents.yaml`).
- For rendering in robosuite: an EGL-capable GPU driver (`MUJOCO_GL=egl`), or `MUJOCO_GL=osmesa` without a GPU.

## 2. Clone and core environment

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks        # strips AI co-author trailers from commits
make install                               # = uv sync (control plane, harness, tests)
make test                                  # GPU-free tests must pass
```

## 3. Simulator and training environments

```bash
# robosuite worker env (Python 3.10). Keep these exact versions.
conda create -y -n robosuite python=3.10
conda run -n robosuite pip install robosuite==1.5.2 "mujoco==3.3.7" "numpy<2" pydantic pyyaml imageio pytest \
    fastapi uvicorn httpx

# lerobot worker env (Python 3.12, CUDA GPU for training)
conda create -y -n lerobot python=3.12
conda run -n lerobot pip install "lerobot>=0.5,<0.6" pytest fastapi uvicorn httpx
```

Check the robosuite env: `MUJOCO_GL=egl conda run -n robosuite python -m pytest -q tests/sim`.

## 4. Start the stack (four terminals, or tmux)

```bash
make dev                # control plane + UI + agent harness  -> http://localhost:8700
make worker-robosuite   # :8711
make worker-lerobot     # :8712 (or start it on the GPU box, see below)
```

The worker lamps in the UI header turn green when a worker answers on its URL (`configs/workers.yaml`).

**Remote GPU box:** clone the repo there, create the `lerobot` env, and run
`conda run -n lerobot python -m orchestrator.worker --name lerobot --capabilities convert,intel,train,policy_server --host 0.0.0.0 --port 8712`.
Then point the `lerobot` entry in `configs/workers.yaml` (or your own file passed with `CATHLAB_WORKERS=...`) at
`http://<gpu-box>:8712`. Use an SSH tunnel (`ssh -N -L 8712:localhost:8712 <gpu-box>`) if the port isn't reachable.

## 5. Try the UI without touching the repo

`make demo-ui` starts the UI against a throwaway git repo with a fake agent. Try AGENT → Free task with paths `configs/`
and check `test -f configs/new.txt`, then Start agent → DIFF → PUSH BRANCH.

## Ports

| Port | Service |
|---|---|
| 8700 | control plane + UI (+ agent harness in dev) |
| 8711 | robosuite worker |
| 8712 | lerobot worker |
| 8713 | isaac worker |
| 8790 | standalone agent worker (`make agent`) |
````

- [ ] **Step 3: Create `docs/HARNESS.md`**

````markdown
# Agent Harness: Team Guide

The harness turns a development task into a **reviewed git branch**, using a coding agent (GitHub Copilot CLI by
default). You pick a task template, fill a few fields, and the harness does the rest:

```
template + your fields → prompt → git worktree on agent/<task>-<id> (from origin/main)
  → agent edits files → path guard (only allowed paths) → acceptance checks → commit (your git identity)
  → you review the DIFF → PUSH BRANCH (then open a PR) or DISCARD
```

Your working folder and `main` are never touched. Every job keeps its prompt, full log and diff in `var/agent/<id>/`.

## Use it from the UI

1. `make dev` and open http://localhost:8700.
2. **PROJECT** tab: anything that isn't ready shows **Build with agent**, and sections with an add-task show
   **+ Add … with agent**.
3. Or go straight to **AGENT** and pick a template on the left.
4. Fill the fields, add extra instructions if needed, choose the agent, then **Start agent**.
5. Watch the live log. When the state is `ready_for_review`:
   - **DIFF** shows the full change;
   - **PROMPT** shows exactly what the agent was told;
   - **PUSH BRANCH** pushes `agent/<task>-<id>`; open a PR on GitHub from it;
   - **DISCARD** deletes the worktree and branch.

## Use it from the terminal

```bash
uv run python -m harness list
uv run python -m harness run add_object --param name=guidewire_hoop --param label="Guidewire hoop" \
    --param shape=cylinder --param radius=0.06 --param length=0.02 --param upright=false --param mass=0.05
uv run python -m harness diff <job_id>
uv run python -m harness push <job_id>        # or: discard <job_id>
```

## Templates

| Template | Use it to | Checks |
|---|---|---|
| `add_object` | Add a pickable object (new versioned spec `cathlab_v2`, scene, task, catalog) | core tests, robosuite tests, expert ≥ 80 % on the new object |
| `extend_dataset` | Generate more episodes for a window of configs, convert, preview, push to HF | dataset + manifest exist |
| `add_variation_axis` | Add a randomization axis (camera jitter, distractors, …) | core + robosuite tests, expert ≥ 80 % |
| `add_sim_backend` | Build an Isaac or MuJoCo backend against `ExpertSim` / `SimBackend` | core tests |
| `add_policy_adapter` | Wrap a new model (adapter + training recipe) | core tests |
| `add_expert` | Add a grasp source such as GraspGen | core + robosuite tests |
| `free_task` | Anything else: your goal, your allowed paths, your check | your command |

To add a template, drop a YAML file in `harness/tasks/` (the file name must equal its `id`).
`tests/harness/test_templates.py` checks that it loads and renders.

## Rules

- Review every diff before pushing. The agent is fast, not infallible.
- Templates never edit `configs/variation/cathlab_v1.yaml`: the published v1 dataset depends on it.
- Commits carry **your** name; the repo hook strips AI co-author trailers (`git config core.hooksPath .githooks`).
- Jobs that need robosuite run checks in the `robosuite` conda env, so set it up first (docs/INSTALL.md).
- Agents: `configs/agents.yaml` (`copilot` default; `claude`, `codex` optional). Log in to the agent CLI on the host
  before starting the control plane.
````

- [ ] **Step 4: Create `CREDITS.md`**

````markdown
# Credits

CathLab Companion builds on these projects. Thank you to their authors. Each is used under its own licence.

| Project | Used for | Link | Licence |
|---|---|---|---|
| robosuite | Simulation framework for the Panda pick task | https://github.com/ARISE-Initiative/robosuite | MIT |
| MuJoCo | Physics engine under robosuite | https://github.com/google-deepmind/mujoco | Apache-2.0 |
| LeRobot | Dataset format (v3), training, policies | https://github.com/huggingface/lerobot | Apache-2.0 |
| SmolVLA | Baseline vision-language-action policy (via LeRobot) | https://huggingface.co/lerobot/smolvla_base | Apache-2.0 |
| ACT | Non-language baseline policy (via LeRobot) | https://github.com/tonyzhaozh/act | MIT |
| NVIDIA Isaac Sim / Isaac Lab | Provided FR3 Duo clinical scene | https://github.com/isaac-sim/IsaacLab | BSD-3-Clause (Isaac Lab); Isaac Sim under NVIDIA licence |
| Hugging Face Hub | Dataset and model hosting | https://huggingface.co | - |
| FastAPI, uvicorn, Starlette | Control plane and worker HTTP APIs | https://github.com/fastapi/fastapi | MIT / BSD-3-Clause |
| pydantic | Configs, contracts, API models | https://github.com/pydantic/pydantic | MIT |
| GitHub Copilot CLI | Default coding agent of the harness | https://github.com/github/copilot-cli | GitHub terms |
| LIBERO | Action/observation conventions (7-D EEF delta, agentview + eye-in-hand) | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT |
| RoboSimStudio | Prompt-to-scene showcase (optional) | https://github.com/robosimstudio/robosimstudio | Apache-2.0 |

Papers and notes:
- Zhou & Zhang, *A tutorial note on collecting simulated data for vision-language-action models*, arXiv:2508.06547
  (no-op filtering, success hold, state recording, validation on aggregation).
- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, arXiv:2506.01844.
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), arXiv:2304.13705.
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, arXiv:2306.03310.
- Zhu et al., *robosuite: A Modular Simulation Framework and Benchmark for Robot Learning*, arXiv:2009.12293.

The Siemens Healthineers hackathon materials (brief, provided scene and datasets) are restricted and are not
redistributed here.
````

- [ ] **Step 5: Add a Control Station section to `README.md`**

Insert directly under the first paragraph (after the loop diagram block):

````markdown
## Control Station (UI + agent harness)

```bash
make install && make test      # core env + GPU-free tests
make dev                       # http://localhost:8700  (UI, catalog, runs, agent harness)
make worker-robosuite          # robosuite worker :8711 (needs the robosuite conda env)
make demo-ui                   # try the UI with a throwaway repo and a fake agent
```

- **PROJECT:** pick simulator, embodiment, objects, expert, variation and model from the catalog. Anything not ready offers **Build with agent**.
- **AGENT:** run a task template with a coding agent (GitHub Copilot CLI by default) in an isolated git worktree, then review the diff and push the branch. See [docs/HARNESS.md](docs/HARNESS.md).
- Install (native, remote GPU worker): [docs/INSTALL.md](docs/INSTALL.md) · Credits: [CREDITS.md](CREDITS.md).
````

- [ ] **Step 6: Extend `AGENTS.md`**

Add these rows to the ownership table:

```markdown
| `orchestrator/`, `harness/`, `frontend/`, `configs/{catalog,workers,agents}.yaml` | Rakesh |
```

Append this section at the end:

````markdown
## Control Station and agent harness

- Control plane: `python -m orchestrator.control` (FastAPI, port 8700) serves `frontend/`, the catalog (`configs/catalog.yaml`), runs/jobs (`var/control.db`) and `/api/agent/*`.
- Workers: `python -m orchestrator.worker --name <env> --capabilities ... --port 87xx` inside each Python env; URLs in `configs/workers.yaml`.
- Harness: task templates in `harness/tasks/*.yaml` (file name = id). To support a new kind of agent task, add a template rather than special-casing code. Agents run in worktrees on `agent/<task>-<id>`, may only change `allowed_paths`, must not commit or push, and every job ends with a human review.
- When you add a catalog item that is not ready, set `harness_task` to an existing template id (`tests/harness/test_templates.py` enforces this).
````

- [ ] **Step 7: Run the full suite and the demo smoke**

Run: `uv run pytest -q`
Expected: all tests pass (the Task 0 tests plus 50 from Tasks 1–9).

Run: `uv run python -m harness list`
Expected: 7 lines, one per template.

- [ ] **Step 8: Commit and publish**

```bash
git add Makefile docs/INSTALL.md docs/HARNESS.md CREDITS.md README.md AGENTS.md
git commit -m "docs: Control Station install, harness team guide, credits; make targets"
git fetch -q origin && git rebase origin/main
uv run pytest -q
git push origin HEAD:main
git push -u origin feat/rakesh-control-station
```

---

## Spec coverage (Plan A)

| Spec item | Task |
|---|---|
| C10 step 0 integration | 0 |
| C4 catalog + runs | 1, 2, 8 |
| C1/C2 control plane + workers (topology) | 3, 4, 8 |
| C5 agent harness (templates, worktree, guard, checks, review gate, pluggable CLIs, CLI) | 5, 6, 7, 9 |
| C6 frontend (PolyScope tabs + tree + footer, stage strip, agent console) | 10 |
| C7 state (SQLite, `var/`, logs) | 1, 2, 3 |
| §8 credits, install, team guide | 11 |
| C3 policy server, §4 stages, eval suite v2, registry/promote/rollout, train, C8 Docker | **Plan B** |
