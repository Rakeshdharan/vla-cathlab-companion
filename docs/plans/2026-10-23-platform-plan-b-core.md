# Platform Plan B-core: Engine SDK, Stage Pipeline, Gates and Agent Brief — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Control Station run a real pipeline (preview → check expert → generate → convert → eval → rollout) for the selected engine from the UI. Give any coding agent the gates and the Agent Brief it needs to add engines, objects, data methods and metrics.

**Architecture:**
- Each simulator is an **engine plugin**: `engines/<id>/engine.yaml` plus a uniform CLI built on `engines/cli_base.py`.
- The control plane builds the CLI command for each stage from `orchestrator/stages.py` and sends it to the engine's worker, or to the `convert` worker. Stage jobs are tracked in the store; logs and output files are proxied to the UI.
- **Stage gates** (`harness/gates/*.yaml`) describe the contract of every stage. `harness/brief.py` assembles a deterministic markdown brief from them for any agent.

**Spec:** [`docs/specs/2026-10-23-multi-engine-platform-design.md`](../specs/2026-10-23-multi-engine-platform-design.md) (P1, P3, P4, P7 eval reference, P9 runner part, P13, P14, P15).
- **Follow-up plans:**
  - **B-loop:** train glue, registry/promote/rollback with Sharan's `serve/`, ASSIST, failure report, MODELS/DATA/REPORT tabs.
  - **B-engines:** MuJoCo and Newton engines, object packages and importers, P6 data-quality rules.

**Tech Stack:**
- Python 3.10+ (the robosuite env is 3.10: no 3.11-only syntax under `engines/`, `sim/`, `dataengine/`), pydantic v2, PyYAML.
- FastAPI and httpx (control plane).
- argparse (engine CLIs); robosuite 1.5.2 / MuJoCo 3.3.7 in the `robosuite` conda env.
- Vanilla ES modules (frontend), pytest.

## Global Constraints

- Commits are authored by the human only. **No `Co-authored-by:` trailers for AI agents.** Conventional Commit messages.
- Never edit `configs/variation/cathlab_v1.yaml` (the published v1 dataset depends on it).
- Core tests (`uv run pytest -q`) stay GPU- and simulator-free. Anything needing robosuite uses `pytest.importorskip("robosuite")` and runs with `MUJOCO_GL=egl conda run -n robosuite python -m pytest ...`.
- Engine CLIs import simulators **lazily** (inside stage functions or wrapped scripts). `python -m engines.<id>.cli --help` must work in the core env and in the `lerobot` env.
- Engine CLI contract:
  - exit 0 on success, non-zero with a **one-line reason on stderr**;
  - progress on stdout;
  - outputs only under `--out`;
  - `--run <run.yaml>` restricts objects through `selection.objects`.
- Never write to `/tmp`; tests use `tmp_path`.
- Data, checkpoints, `.env` and secrets never go into git. Run outputs go under `var/` (git-ignored).
- Every external project you use gets a row in `CREDITS.md`.
- Worker auth header `X-Cathlab-Token` from env `CATHLAB_WORKER_TOKEN`. Control port 8700, robosuite worker 8711, lerobot worker 8712, isaac worker 8713.
- Expected baseline before Task 1: `uv run pytest -q` → **190 passed, 5 skipped**.

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `engines/__init__.py`, `engines/registry.py` | `EngineManifest`, `IO_PROFILES`, `load_engines`, `get_engine`, `status()` | 1 |
| `engines/robosuite/engine.yaml` | robosuite manifest | 1, 4 |
| `contracts/specs.py` | `Backend` literal gains `newton` | 1 |
| `engines/cli_base.py` | `Flag`, `Stage`, `StageContext`, `run_cli` (argparse, run.yaml, errors, dry-run) | 2 |
| `engines/robosuite/cli.py` | Stages preview, check-expert, generate, convert (script wrappers), eval, rollout | 3, 4 |
| `engines/policies.py` | `make_policy("random" \| "url:<u>")` | 4 |
| `engines/expert_policy.py` | `ExpertPolicy`: the scripted expert as a `PolicyAdapter` | 4 |
| `engines/robosuite/evaluate.py` | `run_eval`, `run_rollout`, `run_episode`, `save_video`, `load_suite` | 4 |
| `configs/eval/suite_v2.yaml` | 3 objects × 5 tiers × 4 seeds | 4 |
| `tests/engines/test_conformance.py` | Per-engine conformance (skips without the simulator) | 4 |
| `orchestrator/stages.py` | `StageDef`, `StageParam`, `STAGES`, `stage_argv` | 5 |
| `orchestrator/pipeline.py` | `Pipeline` (check, start, refresh, log, cancel, files), `worker_request` | 5 |
| `orchestrator/control.py` | Stage, engine, job-log and file endpoints | 5 |
| `tests/engines/fake/engine.yaml`, `tests/engines/fake_engine_cli.py` | Fake engine for pipeline tests | 5 |
| `frontend/js/pipeline.js`, `frontend/index.html`, `frontend/js/app.js`, `frontend/css/station.css` | PIPELINE tab | 6 |
| `harness/gates.py`, `harness/gates/*.yaml` (12) | Stage gates | 7 |
| `harness/brief.py`, `harness/api.py`, `harness/__main__.py` | Agent Brief: function, API, CLI | 7 |
| `harness/model.py`, `harness/tasks/*.yaml` | `TaskTemplate.gate`; 5 new templates | 7 |
| `frontend/js/brief.js`, `frontend/index.html`, `configs/catalog.yaml` | Brief drawer on every tab, `newton` catalog item | 8 |
| `docs/EXTENDING.md`, `engines/README.md`, `AGENTS.md`, `.github/instructions/*.instructions.md`, `.github/skills/cathlab-extend/SKILL.md`, `docs/HARNESS.md`, `docs/INSTALL.md`, `README.md`, `Makefile`, `docs/handoffs/2026-10-23-isaac-engine-session.md` | Repo-wide agent instructions and docs | 9 |

---

### Task 1: Engine manifest and registry

**Files:**
- Create: `engines/__init__.py`, `engines/registry.py`, `engines/robosuite/__init__.py`, `engines/robosuite/engine.yaml`, `tests/engines/__init__.py`, `tests/engines/test_registry.py`
- Modify: `contracts/specs.py` (the `Backend` literal)

**Interfaces:**
- Produces:
  - `engines.registry.ROOT: Path`, `ENGINES_DIR: Path`, `Status = Literal["ready","in_progress","planned"]`;
  - `IOProfile(id, action_dim, fps, obs_shapes: dict[str, tuple[int,...]], description)`, `IO_PROFILES: dict[str, IOProfile]` (`panda_libero`, `fr3_duo_16d`);
  - `EngineManifest` (fields below) with `.io_profile` and `.status(capability: str) -> Status`;
  - `EngineError(ValueError)`;
  - `load_manifest(path) -> EngineManifest`, `load_engines(engines_dir=ENGINES_DIR) -> dict[str, EngineManifest]`, `get_engine(engine_id, engines_dir=ENGINES_DIR) -> EngineManifest`.

- [ ] **Step 1: Write the failing tests** in `tests/engines/test_registry.py` (also create an empty `tests/engines/__init__.py`):

```python
import pytest
from contracts.specs import SceneSpec
from engines.registry import IO_PROFILES, EngineError, get_engine, load_engines, load_manifest

MANIFEST = """id: fake
label: Fake engine
env: {kind: conda, name: fake}
worker: fake
embodiments: [panda]
io: panda_libero
variation: configs/variation/cathlab_v1.yaml
task: configs/tasks/cathlab_pick.yaml
capabilities:
  preview: ready
  expert: {analytic: ready, graspgen: planned}
  generate: in_progress
entry: engines.fake.cli
"""


def write(tmp_path, text=MANIFEST, folder="fake"):
    d = tmp_path / folder
    d.mkdir()
    (d / "engine.yaml").write_text(text)
    return d / "engine.yaml"


def test_repo_engines_load_and_robosuite_is_ready_for_data():
    rs = load_engines()["robosuite"]
    assert rs.worker == "robosuite" and rs.entry == "engines.robosuite.cli"
    assert rs.io_profile.action_dim == 7 and rs.io_profile.obs_shapes["state"] == (8,)
    for cap in ("preview", "expert.analytic", "generate", "convert"):
        assert rs.status(cap) == "ready", cap


def test_status_handles_groups_and_unknown(tmp_path):
    m = load_manifest(write(tmp_path))
    assert m.status("preview") == "ready"
    assert m.status("expert.analytic") == "ready" and m.status("expert.graspgen") == "planned"
    assert m.status("expert") == "ready"          # a group reports its best child
    assert m.status("generate") == "in_progress"
    assert m.status("rollout") == "planned" and m.status("expert.mimicgen") == "planned"


def test_id_must_match_folder(tmp_path):
    with pytest.raises(EngineError, match="folder"):
        load_manifest(write(tmp_path, folder="other"))


def test_unknown_io_profile_rejected(tmp_path):
    with pytest.raises(ValueError, match="io profile"):
        load_manifest(write(tmp_path, MANIFEST.replace("panda_libero", "nope")))


def test_get_engine_unknown(tmp_path):
    with pytest.raises(EngineError, match="no engine plugin 'mujoco'"):
        get_engine("mujoco", tmp_path)


def test_io_profiles_and_newton_backend_literal():
    assert IO_PROFILES["fr3_duo_16d"].action_dim == 16 and IO_PROFILES["fr3_duo_16d"].fps == 30
    assert SceneSpec(scene_id="s", backend="newton", asset="", objects={}, cameras=[]).backend == "newton"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/engines/test_registry.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines'`.

- [ ] **Step 3: Implement.** Create `engines/__init__.py`:

```python
"""Engine plugins: one folder per simulator with engine.yaml + cli.py. See engines/README.md and docs/EXTENDING.md."""
```

`engines/registry.py`:

```python
"""Engine plugin manifests (engines/<id>/engine.yaml): load, validate, query capabilities. No simulator imports."""
from pathlib import Path
from typing import Literal, Union
import yaml
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
ENGINES_DIR = ROOT / "engines"
Status = Literal["ready", "in_progress", "planned"]
STATUS_RANK = {"planned": 0, "in_progress": 1, "ready": 2}


class IOProfile(BaseModel):
    id: str
    action_dim: int
    fps: int
    obs_shapes: dict[str, tuple[int, ...]] = Field(default_factory=dict)  # empty: not checked by conformance
    description: str = ""


IO_PROFILES: dict[str, IOProfile] = {
    "panda_libero": IOProfile(
        id="panda_libero", action_dim=7, fps=20,
        obs_shapes={"state": (8,), "agentview": (256, 256, 3), "eye_in_hand": (256, 256, 3)},
        description="7-D normalized EEF delta [dpos, drot, gripper], 8-D state, agentview + eye_in_hand 256², 20 Hz"),
    "fr3_duo_16d": IOProfile(
        id="fr3_duo_16d", action_dim=16, fps=30,
        description="Siemens FR3 Duo schema: 16-D joint targets, 30 Hz (see the Isaac engine handoff)"),
}


class EngineEnv(BaseModel):
    kind: Literal["conda", "venv", "command"]
    name: str | None = None     # conda env name or venv path
    python: str | None = None   # interpreter command for kind=command, e.g. "${ISAAC_PYTHON}"


class Credit(BaseModel):
    name: str
    url: str = ""
    license: str = ""


class EngineManifest(BaseModel):
    id: str
    label: str
    env: EngineEnv
    worker: str                  # worker name in configs/workers.yaml that runs this engine's stages
    embodiments: list[str]
    io: str
    variation: str               # default variation spec, repo-relative
    task: str                    # task spec, repo-relative
    scene_asset_env: str | None = None
    capabilities: dict[str, Union[Status, dict[str, Status]]]
    entry: str                   # python -m <entry> <stage> ...
    backend: str | None = None   # "module:Class" implementing contracts.SimBackend (conformance test)
    credits: list[Credit] = Field(default_factory=list)

    @field_validator("io")
    @classmethod
    def _known_io(cls, v: str) -> str:
        if v not in IO_PROFILES:
            raise ValueError(f"unknown io profile '{v}'; expected one of {sorted(IO_PROFILES)}")
        return v

    @property
    def io_profile(self) -> IOProfile:
        return IO_PROFILES[self.io]

    def status(self, capability: str) -> Status:
        """`generate` or dotted `expert.analytic`. A group without a key reports its best child. Unknown: planned."""
        key, _, sub = capability.partition(".")
        value = self.capabilities.get(key)
        if value is None:
            return "planned"
        if isinstance(value, str):
            return value
        if sub:
            return value.get(sub, "planned")
        return max(value.values(), key=STATUS_RANK.__getitem__, default="planned")


class EngineError(ValueError):
    pass


def load_manifest(path: str | Path) -> EngineManifest:
    path = Path(path)
    manifest = EngineManifest.model_validate(yaml.safe_load(path.read_text()))
    if manifest.id != path.parent.name:
        raise EngineError(f"{path}: id '{manifest.id}' must equal its folder name '{path.parent.name}'")
    return manifest


def load_engines(engines_dir: str | Path = ENGINES_DIR) -> dict[str, EngineManifest]:
    manifests = (load_manifest(p) for p in sorted(Path(engines_dir).glob("*/engine.yaml")))
    return {m.id: m for m in manifests}


def get_engine(engine_id: str, engines_dir: str | Path = ENGINES_DIR) -> EngineManifest:
    path = Path(engines_dir) / engine_id / "engine.yaml"
    if not path.exists():
        raise EngineError(f"no engine plugin '{engine_id}' (expected {path})")
    return load_manifest(path)
```

Create `engines/robosuite/__init__.py` (empty) and `engines/robosuite/engine.yaml`:

```yaml
# robosuite engine plugin. Contract: docs/EXTENDING.md and harness/gates/engine.yaml.
id: robosuite
label: "robosuite 1.5 (MuJoCo 3.3): Panda, cathlab table scene"
env: {kind: conda, name: robosuite}
worker: robosuite
embodiments: [panda]
io: panda_libero
variation: configs/variation/cathlab_v1.yaml
task: configs/tasks/cathlab_pick.yaml
capabilities:                    # ready | in_progress | planned
  preview: ready
  objects: {primitive: ready, mjcf: planned}
  randomize: ready
  expert: {analytic: ready, graspgen: planned}
  generate: ready
  convert: ready
  eval: in_progress              # flipped to ready in Task 4
  rollout: in_progress
entry: engines.robosuite.cli
backend: "sim.robosuite.backend:RobosuiteBackend"
credits:
  - {name: robosuite, url: "https://github.com/ARISE-Initiative/robosuite", license: MIT}
  - {name: MuJoCo, url: "https://github.com/google-deepmind/mujoco", license: Apache-2.0}
```

In `contracts/specs.py`, change the `Backend` line to:

```python
Backend = Literal["dummy", "robosuite", "isaac", "mujoco", "newton", "robosimstudio"]
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/engines/test_registry.py` → 6 passed. Then `uv run pytest -q` → 196 passed, 5 skipped.

- [ ] **Step 5: Commit**

```bash
git add engines tests/engines contracts/specs.py
git commit -m "feat(engines): engine manifest registry with io profiles; robosuite manifest; newton backend literal"
```

---

### Task 2: Uniform engine CLI base

**Files:**
- Create: `engines/cli_base.py`, `tests/engines/test_cli_base.py`

**Interfaces:**
- Consumes: `engines.registry.ROOT`, `EngineManifest`, `get_engine` (Task 1).
- Produces:
  - `Flag(name, type=str, default=None, help="", action=None, required=False)`. `action` is `"append"` or `"store_true"`. The argparse dest is the name without `--`, with `-` replaced by `_`.
  - `StageContext(manifest, out: Path, run: dict | None = None, dry_run=False, root=ROOT)`, with:
    - `.selection -> dict`;
    - `.objects -> list[str]` (empty without a run);
    - `.path(rel) -> str` (absolute path under `root`);
    - `.run_script(argv, env=None) -> int`, which runs `[sys.executable, *argv]` with `cwd=root`. Under dry-run it prints `json.dumps(cmd)` on one line and returns 0.
  - `Stage(fn: Callable[[Namespace, StageContext], int | None], help="", flags: tuple[Flag, ...] = ())`.
  - `build_parser(engine_id, stages) -> ArgumentParser`.
  - `run_cli(engine_id, stages: dict[str, Stage], argv=None, manifest=None) -> int`.

- [ ] **Step 1: Write the failing tests** in `tests/engines/test_cli_base.py`:

```python
import json
import sys
import pytest
from engines.cli_base import Flag, Stage, run_cli
from engines.registry import get_engine

RS = get_engine("robosuite")


def _write(args, ctx):
    (ctx.out / "ok.txt").write_text(f"n={args.n} objects={','.join(ctx.objects)}")


def _script(args, ctx):
    return ctx.run_script(["-c", "print('hi')"])


def _boom(args, ctx):
    raise RuntimeError("no simulator here")


STAGES = {
    "write": Stage(_write, flags=(Flag("--n", int, 3),)),
    "script": Stage(_script),
    "boom": Stage(_boom),
    "need": Stage(lambda a, c: None, flags=(Flag("--dataset", required=True), Flag("--param", action="append"))),
}


def test_stage_gets_flags_out_and_run_objects(tmp_path):
    run = tmp_path / "run.yaml"
    run.write_text("id: r1\nselection: {objects: [vial, syringe]}\n")
    out = tmp_path / "out"
    assert run_cli("robosuite", STAGES, ["write", "--out", str(out), "--run", str(run), "--n", "5"], manifest=RS) == 0
    assert (out / "ok.txt").read_text() == "n=5 objects=vial,syringe"


def test_dry_run_prints_argv_and_creates_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert run_cli("robosuite", STAGES, ["script", "--out", str(out), "--dry-run"], manifest=RS) == 0
    cmd = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert cmd == [sys.executable, "-c", "print('hi')"] and not out.exists()


def test_failure_is_one_line_on_stderr(tmp_path, capsys):
    assert run_cli("robosuite", STAGES, ["boom", "--out", str(tmp_path)], manifest=RS) == 1
    assert capsys.readouterr().err.strip() == "robosuite boom failed: no simulator here"


def test_script_exit_code_propagates(tmp_path, capsys):
    stages = {"bad": Stage(lambda a, c: c.run_script(["-c", "raise SystemExit(3)"]))}
    assert run_cli("robosuite", stages, ["bad", "--out", str(tmp_path)], manifest=RS) == 3
    assert "robosuite bad failed: exit code 3" in capsys.readouterr().err


def test_required_and_unknown_are_usage_errors(tmp_path):
    for argv in (["nope", "--out", str(tmp_path)], ["need", "--out", str(tmp_path)]):
        with pytest.raises(SystemExit) as e:
            run_cli("robosuite", STAGES, argv, manifest=RS)
        assert e.value.code == 2


def test_append_flag_collects_values(tmp_path):
    seen = {}
    stages = {"s": Stage(lambda a, c: seen.update(p=a.param), flags=(Flag("--param", action="append"),))}
    run_cli("robosuite", stages, ["s", "--out", str(tmp_path), "--param", "a=1", "--param", "b=2"], manifest=RS)
    assert seen["p"] == ["a=1", "b=2"]
    run_cli("robosuite", stages, ["s", "--out", str(tmp_path)], manifest=RS)
    assert seen["p"] == []
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/engines/test_cli_base.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines.cli_base'`.

- [ ] **Step 3: Implement** `engines/cli_base.py`:

```python
"""Uniform engine CLI. Each engine's cli.py declares its stages; this module does argparse, run.yaml and errors.

    python -m engines.<id>.cli <stage> --out DIR [--run var/runs/<id>/run.yaml] [stage flags] [--dry-run]

Contract (docs/EXTENDING.md): exit 0 on success; non-zero with a one-line reason on stderr; progress on stdout;
outputs only under --out. --dry-run prints each command a stage would run (one JSON argv per line), runs nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from engines.registry import ROOT, EngineManifest, get_engine


@dataclass(frozen=True)
class Flag:
    name: str                              # "--n"
    type: Callable[[str], Any] = str
    default: Any = None
    help: str = ""
    action: str | None = None              # "append" | "store_true"
    required: bool = False

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        if self.action == "store_true":
            parser.add_argument(self.name, action="store_true", help=self.help)
        elif self.action == "append":
            parser.add_argument(self.name, action="append", default=None, help=self.help)
        else:
            parser.add_argument(self.name, type=self.type, default=self.default, help=self.help,
                                required=self.required)


@dataclass
class StageContext:
    manifest: EngineManifest
    out: Path
    run: dict | None = None
    dry_run: bool = False
    root: Path = ROOT

    @property
    def selection(self) -> dict:
        return (self.run or {}).get("selection") or {}

    @property
    def objects(self) -> list[str]:
        """Objects of the run's selection; empty without a run (the stage default applies)."""
        objs = self.selection.get("objects") or []
        return [objs] if isinstance(objs, str) else list(objs)

    def path(self, rel: str) -> str:
        p = Path(rel)
        return str(p if p.is_absolute() else self.root / p)

    def run_script(self, argv: list, env: dict[str, str] | None = None) -> int:
        """Run [sys.executable, *argv] from the repo root; with --dry-run only print the argv as JSON."""
        cmd = [sys.executable, *[str(a) for a in argv]]
        if self.dry_run:
            print(json.dumps(cmd), flush=True)
            return 0
        print("$ " + " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=self.root, env={**os.environ, **(env or {})}).returncode


@dataclass(frozen=True)
class Stage:
    fn: Callable[[argparse.Namespace, StageContext], "int | None"]
    help: str = ""
    flags: tuple = ()


def build_parser(engine_id: str, stages: dict[str, Stage]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m engines.{engine_id}.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, stage in stages.items():
        p = sub.add_parser(name, help=stage.help, description=stage.help)
        p.add_argument("--out", required=True, help="output folder (created)")
        p.add_argument("--run", help="Control Station run snapshot (run.yaml); selection.objects restricts objects")
        p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
        for flag in stage.flags:
            flag.add_to(p)
    return ap


def run_cli(engine_id: str, stages: dict[str, Stage], argv: list[str] | None = None,
            manifest: EngineManifest | None = None) -> int:
    args = build_parser(engine_id, stages).parse_args(argv)
    for flag in stages[args.stage].flags:      # append flags default to an empty list, never a shared one
        dest = flag.name.lstrip("-").replace("-", "_")
        if flag.action == "append" and getattr(args, dest) is None:
            setattr(args, dest, [])
    try:
        manifest = manifest or get_engine(engine_id)
        out = Path(args.out)
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        run = yaml.safe_load(Path(args.run).read_text()) if args.run else None
        rc = stages[args.stage].fn(args, StageContext(manifest, out, run, args.dry_run))
        rc = 0 if rc is None else int(rc)
    except Exception as e:  # noqa: BLE001 - engine contract: one-line reason on stderr, non-zero exit
        traceback.print_exc(file=sys.stdout)
        print(f"{engine_id} {args.stage} failed: {e}", file=sys.stderr)
        return 1
    if rc != 0:
        print(f"{engine_id} {args.stage} failed: exit code {rc}", file=sys.stderr)
    return rc
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/engines` → all pass.

- [ ] **Step 5: Commit**

```bash
git add engines/cli_base.py tests/engines/test_cli_base.py
git commit -m "feat(engines): uniform engine CLI base (flags, run.yaml objects, dry-run, one-line errors)"
```

---

### Task 3: robosuite engine CLI (preview, check-expert, generate, convert)

**Files:**
- Create: `engines/robosuite/cli.py`, `tests/engines/test_robosuite_cli.py`

**Interfaces:**
- Consumes: `Flag`, `Stage`, `StageContext`, `run_cli` (Task 2). The existing scripts:
  - `scripts/robosuite_variation_grid.py --out --n --spec`;
  - `scripts/robosuite_expert_check.py --n --split --objects --param --spec --json`;
  - `scripts/robosuite_generate.py --split --start --n --out --shard --num-shards --dataset-id --version --created-by --spec --task`;
  - `scripts/npz_to_lerobot.py --dataset --root --repo-id --fps`.
- Produces:
  - `engines.robosuite.cli.STAGES: dict[str, Stage]` with keys `preview`, `check-expert`, `generate`, `convert`. Task 4 adds `eval` and `rollout`.
  - `main(argv=None) -> int`.
  - Outputs in `--out`:
    - preview → `grid.png`;
    - check-expert → `expert_check.json`;
    - generate → `raw/shard_<i>-of-<k>/…`;
    - convert → `lerobot/`.

- [ ] **Step 1: Write the failing tests** in `tests/engines/test_robosuite_cli.py` (core env, dry-run only):

```python
import json
import sys
import pytest
from engines.registry import ROOT
from engines.robosuite.cli import main

SPEC = str(ROOT / "configs/variation/cathlab_v1.yaml")


def argv_of(capsys, *args):
    assert main([*args, "--dry-run"]) == 0
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def val(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_preview(tmp_path, capsys):
    cmd = argv_of(capsys, "preview", "--out", str(tmp_path), "--n", "4")
    assert cmd[:2] == [sys.executable, "scripts/robosuite_variation_grid.py"]
    assert val(cmd, "--out") == str(tmp_path / "grid.png") and val(cmd, "--n") == "4" and val(cmd, "--spec") == SPEC


def test_check_expert_uses_run_objects_and_params(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial]}\n")
    cmd = argv_of(capsys, "check-expert", "--out", str(tmp_path), "--run", str(run), "--n", "5",
                  "--param", "pos_tol=0.01")
    assert cmd[1] == "scripts/robosuite_expert_check.py"
    assert val(cmd, "--objects") == "vial" and val(cmd, "--param") == "pos_tol=0.01" and val(cmd, "--n") == "5"
    assert val(cmd, "--json") == str(tmp_path / "expert_check.json")


def test_check_expert_without_run_checks_all_objects(tmp_path, capsys):
    cmd = argv_of(capsys, "check-expert", "--out", str(tmp_path))
    assert "--objects" not in cmd and val(cmd, "--n") == "20"


def test_generate_window_and_shard(tmp_path, capsys):
    cmd = argv_of(capsys, "generate", "--out", str(tmp_path), "--start", "10", "--n", "6", "--shard", "1",
                  "--num-shards", "2")
    assert cmd[1] == "scripts/robosuite_generate.py"
    expected = {"--start": "10", "--n": "6", "--shard": "1", "--num-shards": "2", "--out": str(tmp_path),
                "--version": "v1", "--dataset-id": "cathlab/pick_robosuite", "--split": "train", "--spec": SPEC,
                "--task": str(ROOT / "configs/tasks/cathlab_pick.yaml")}
    assert {k: val(cmd, k) for k in expected} == expected


def test_convert(tmp_path, capsys):
    cmd = argv_of(capsys, "convert", "--out", str(tmp_path), "--dataset", "var/runs/r1/generate/j1")
    assert cmd[1] == "scripts/npz_to_lerobot.py"
    assert val(cmd, "--dataset") == "var/runs/r1/generate/j1" and val(cmd, "--root") == str(tmp_path / "lerobot")
    assert val(cmd, "--fps") == "20" and "--repo-id" not in cmd
    with pytest.raises(SystemExit):
        main(["convert", "--out", str(tmp_path), "--dry-run"])          # --dataset is required


def test_help_works_without_robosuite(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "check-expert" in capsys.readouterr().out
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/engines/test_robosuite_cli.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines.robosuite.cli'`.

- [ ] **Step 3: Implement** `engines/robosuite/cli.py`:

```python
"""robosuite engine CLI: the robosuite scripts behind the uniform engine contract (engines/cli_base.py).

    MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli preview --out var/smoke/preview
    conda run -n lerobot python -m engines.robosuite.cli convert --dataset <generate --out> --out <dir>

Only stage functions live here; robosuite and lerobot are imported by the wrapped scripts or lazily.
"""
from engines.cli_base import Flag, Stage, run_cli

ENGINE = "robosuite"
SPEC = Flag("--spec", help="variation spec (default: the manifest's variation)")
OBJECTS = Flag("--objects", help="comma-separated objects (default: the run's objects, else all)")


def _spec(args, ctx) -> str:
    return args.spec or ctx.path(ctx.manifest.variation)


def _objects(args, ctx) -> str | None:
    return args.objects or (",".join(ctx.objects) if ctx.objects else None)


def preview(args, ctx):
    return ctx.run_script(["scripts/robosuite_variation_grid.py", "--out", ctx.out / "grid.png", "--n", args.n,
                           "--spec", _spec(args, ctx)])


def check_expert(args, ctx):
    argv = ["scripts/robosuite_expert_check.py", "--n", args.n, "--split", args.split, "--spec", _spec(args, ctx),
            "--json", ctx.out / "expert_check.json"]
    objects = _objects(args, ctx)
    if objects:
        argv += ["--objects", objects]
    for p in args.param:
        argv += ["--param", p]
    return ctx.run_script(argv)


def generate(args, ctx):
    argv = ["scripts/robosuite_generate.py", "--split", args.split, "--start", args.start, "--n", args.n,
            "--out", ctx.out, "--shard", args.shard, "--num-shards", args.num_shards,
            "--dataset-id", args.dataset_id, "--version", args.version,
            "--spec", _spec(args, ctx), "--task", ctx.path(ctx.manifest.task)]
    if args.created_by:
        argv += ["--created-by", args.created_by]
    return ctx.run_script(argv)


def convert(args, ctx):
    argv = ["scripts/npz_to_lerobot.py", "--dataset", args.dataset, "--root", ctx.out / "lerobot",
            "--fps", ctx.manifest.io_profile.fps]
    if args.repo_id:
        argv += ["--repo-id", args.repo_id]
    return ctx.run_script(argv)


STAGES = {
    "preview": Stage(preview, "grid.png of n train configs + 1 all-held-out (agentview after settle)",
                     (Flag("--n", int, 8, "train configs in the grid"), SPEC)),
    "check-expert": Stage(check_expert, "scripted expert success per object -> expert_check.json",
                          (Flag("--n", int, 20, "episodes per object"), Flag("--split", default="train"), OBJECTS,
                           Flag("--param", action="append", help="ExpertParams override k=v (repeatable)"), SPEC)),
    "generate": Stage(generate, "successful expert episodes -> raw npz shards in --out/raw/ (all objects, balanced)",
                      (Flag("--split", default="train"), Flag("--start", int, 0, "first config of the window"),
                       Flag("--n", int, 30, "configs in the window (all shards)"), Flag("--shard", int, 0),
                       Flag("--num-shards", int, 1), Flag("--dataset-id", default="cathlab/pick_robosuite"),
                       Flag("--version", default="v1"), Flag("--created-by"), SPEC)),
    "convert": Stage(convert, "raw shards -> LeRobot v3 dataset in --out/lerobot (run in the lerobot env)",
                     (Flag("--dataset", required=True, help="the generate stage's --out folder"),
                      Flag("--repo-id", help="default: dataset_id from the shard manifests"))),
}


def main(argv: list[str] | None = None) -> int:
    return run_cli(ENGINE, STAGES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/engines` → all pass.

- [ ] **Step 5: Real smoke in the robosuite env** (it must already have `fastapi`/`pydantic`/`pyyaml`; it does):

```bash
MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m engines.robosuite.cli check-expert \
    --out var/smoke/check_expert --n 2 --objects vial
cat var/smoke/check_expert/expert_check.json      # per_object.vial.attempted == 2
MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m engines.robosuite.cli preview --out var/smoke/preview --n 2
ls var/smoke/preview/grid.png
```

Expected: both exit 0 and the files exist. `var/` is git-ignored.

- [ ] **Step 6: Commit**

```bash
git add engines/robosuite/cli.py tests/engines/test_robosuite_cli.py
git commit -m "feat(engines): robosuite engine CLI wrapping preview, expert check, generate and convert"
```

---

### Task 4: robosuite eval and rollout, expert as a policy, suite v2, engine conformance

**Files:**
- Create:
  - `engines/policies.py`, `engines/expert_policy.py`, `engines/robosuite/evaluate.py`;
  - `configs/eval/suite_v2.yaml`;
  - `tests/engines/test_policies.py`, `tests/engines/test_robosuite_eval.py`, `tests/engines/test_conformance.py`.
- Modify: `engines/robosuite/cli.py` (add `eval` and `rollout`), `engines/robosuite/engine.yaml` (`eval: ready`, `rollout: ready`).

**Interfaces:**
- Consumes:
  - `contracts`: `STAGES`, `EpisodeResult`, `EvalReport`, `PolicyAdapter`, `PolicyMeta`, `RandomPolicy`, `TaskSpec`, `load_yaml`;
  - `dataengine.expert`: `ExpertParams`, `ExpertSim`, `_reached`, `plan_waypoints`, `waypoint_action`;
  - `dataengine.variation`: `TIERS`, `VariationSpec`, `EpisodeConfig`;
  - `sim.robosuite.backend.RobosuiteBackend`. Its `reset(scene, seed, target_object)` ignores `scene` and calls `reset_config(cfg)`; it exposes `.sim` (`RobosuiteExpertSim`) and a settable `.tier`. Obs are `state` (8,), `agentview` and `eye_in_hand` (256, 256, 3).
  - Sharan's future `serve.policy_client.PolicyClient(url)`, with `.load()` waiting for readiness.
- Produces:
  - `engines.policies.make_policy(spec: str, action_dim: int, seed: int = 0) -> PolicyAdapter`, for `random` and `url:<u>`. It raises `RuntimeError` when `serve.policy_client` is missing and `ValueError` when the spec is unknown.
  - `engines.expert_policy.ExpertPolicy(sim, spec, params=None, control_hz=20.0)`, with `.begin(cfg)` and a `predict` that returns `(1, 7)` float32.
  - `engines.robosuite.evaluate`:
    - `load_suite(path) -> dict`;
    - `run_episode(policy, backend, task, obj, seed, max_steps, instruction=None, frames=None) -> EpisodeResult`;
    - `save_video(frames, stem: Path, fps) -> Path`;
    - `run_eval(suite_path, policy_arg, out, spec_path, task_path, model_version=None, objects=None, tiers=None, max_seeds=None, video="failures") -> EvalReport`. It writes `out/eval_report.json` and `out/videos/*`.
    - `run_rollout(policy_arg, out, spec_path, task_path, obj, seed=5000, tier="in_distribution", instruction=None, max_steps=400) -> dict`. It writes `out/rollout.json` and `out/rollout.{mp4|gif}`.
  - CLI stages:
    - `eval`: `--suite --policy --model-version --tiers --max-seeds --video --objects --spec`;
    - `rollout`: `--policy --object --seed --tier --instruction --max-steps --spec`.
  - `rollout.json` keys: `success, steps, failed_stage, latency_ms_p50, latency_ms_p95, object, tier, seed, instruction, policy, video`.

- [ ] **Step 1: Write the failing core tests** in `tests/engines/test_policies.py`:

```python
import sys
import numpy as np
import pytest
from dataengine.fake_sim import FakeExpertSim
from dataengine.variation import TIERS, VariationSpec, eval_configs
from engines.expert_policy import ExpertPolicy
from engines.policies import make_policy
from engines.registry import ROOT

SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


def test_random_policy_is_one_step_chunks():
    a = make_policy("random", action_dim=7).predict({}, "Hand me the vial")
    assert a.shape == (1, 7) and np.all(np.abs(a) <= 1)


def test_url_policy_without_serve_is_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "serve.policy_client", None)   # import -> ImportError
    with pytest.raises(RuntimeError, match="policy server client not available"):
        make_policy("url:http://127.0.0.1:8720", action_dim=7)


def test_unknown_policy():
    with pytest.raises(ValueError, match="unknown --policy"):
        make_policy("smolvla", action_dim=7)


@pytest.mark.parametrize("cfg", eval_configs(SPEC, "in_distribution", [5000]), ids=lambda c: c.object)
def test_expert_policy_succeeds_in_fake_sim(cfg):
    sim = FakeExpertSim(SPEC)
    policy = ExpertPolicy(sim, SPEC)
    obs = sim.reset(cfg)
    policy.begin(cfg)
    info = None
    for _ in range(400):
        action = policy.predict(obs, "")
        assert action.shape == (1, 7) and action.dtype == np.float32
        obs, info = sim.step(action[0])
        if info.success:
            break
    assert info.success


def test_expert_policy_requires_begin():
    with pytest.raises(RuntimeError, match="begin"):
        ExpertPolicy(FakeExpertSim(SPEC), SPEC).predict({}, "")


def test_suite_v2_is_valid():
    from engines.robosuite.evaluate import load_suite
    suite = load_suite(ROOT / "configs/eval/suite_v2.yaml")
    assert suite["tiers"] == list(TIERS) and suite["objects"] == ["bottle", "vial", "syringe"]
    assert len(suite["seeds"]) == 4 and min(suite["seeds"]) >= 5000
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/engines/test_policies.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'engines.expert_policy'`.

- [ ] **Step 3: Implement the policies and the suite.** `engines/policies.py`:

```python
"""Policy factory for engine eval/rollout: --policy random | url:<policy-server-url> (expert: engine-specific)."""
from contracts import PolicyAdapter, RandomPolicy


def make_policy(spec: str, action_dim: int, seed: int = 0) -> PolicyAdapter:
    """`expert` needs simulator state, so each engine builds engines.expert_policy.ExpertPolicy itself."""
    if spec == "random":
        return RandomPolicy(action_dim=action_dim, chunk_size=1, seed=seed)
    if spec.startswith("url:"):
        try:
            from serve.policy_client import PolicyClient
        except ImportError as e:
            raise RuntimeError("policy server client not available yet (serve/policy_client.py, Sharan's S2)") from e
        return PolicyClient(spec[len("url:"):])
    raise ValueError(f"unknown --policy '{spec}': use expert, random or url:<policy-server-url>")
```

`engines/expert_policy.py`:

```python
"""The scripted waypoint expert as a PolicyAdapter: the reference row of every eval, in any engine.

It reads poses from the engine's ExpertSim, so the engine calls `begin(cfg)` after every reset.
"""
import numpy as np
from contracts import PolicyAdapter, PolicyMeta
from dataengine.expert import ExpertParams, ExpertSim, _reached, plan_waypoints, waypoint_action
from dataengine.variation import EpisodeConfig, VariationSpec


class ExpertPolicy(PolicyAdapter):
    def __init__(self, sim: ExpertSim, spec: VariationSpec, params: ExpertParams | None = None,
                 control_hz: float = 20.0):
        self.sim, self.spec = sim, spec
        self.params = params or ExpertParams()
        self._meta = PolicyMeta(name="expert", params_m=0.0, chunk_size=1, control_hz=control_hz, action_dim=7)
        self._waypoints: list = []
        self._idx = self._held = 0

    def load(self, ckpt: str = "") -> None:
        pass

    def begin(self, cfg: EpisodeConfig) -> None:
        self._waypoints = plan_waypoints(self.spec, cfg.object, self.sim.get_object_pose(cfg.object), self.params,
                                         ee_pose=self.sim.get_ee_pose())
        self._idx = self._held = 0

    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        if not self._waypoints:
            raise RuntimeError("ExpertPolicy.begin(cfg) must be called after every reset")
        ee = self.sim.get_ee_pose()
        wp = self._waypoints[self._idx]
        # Same advance rule as dataengine.expert.run_expert_episode; the last waypoint (retreat) is held.
        if self._idx < len(self._waypoints) - 1 and _reached(ee, wp, self.params):
            if self._held >= wp.hold:
                self._idx, self._held = self._idx + 1, 0
                wp = self._waypoints[self._idx]
            else:
                self._held += 1
        return waypoint_action(ee, wp, self.params)[None, :]

    def meta(self) -> PolicyMeta:
        return self._meta
```

`configs/eval/suite_v2.yaml`:

```yaml
# Eval suite v2 (spec P9): 3 objects × 5 variation tiers × 4 seeds = 60 episodes per engine and model.
# Seeds >= 5000 never overlap the training grid; the same seed is the same episode in every engine.
suite_id: suite_v2
task: configs/tasks/cathlab_pick.yaml
objects: [bottle, vial, syringe]
tiers: [in_distribution, held_out_layout, held_out_obj_color, held_out_table_color, all_held_out]
seeds: [5000, 5001, 5002, 5003]
max_steps: 400
latency_budget_ms_p95: 200.0
```

`engines/robosuite/evaluate.py` (`load_suite` must import without robosuite):

```python
"""Closed-loop eval and live rollout in robosuite (run in the `robosuite` env). robosuite is imported lazily."""
import json
import time
from pathlib import Path
import numpy as np
import yaml
from contracts import STAGES, EpisodeResult, EvalReport, TaskSpec, load_yaml
from dataengine.variation import TIERS, VariationSpec

VIDEO_MODES = ("none", "failures", "all")
FPS = 20
FRAME_EVERY = 2          # keep every 2nd frame in videos


def load_suite(path) -> dict:
    suite = yaml.safe_load(Path(path).read_text())
    bad = [t for t in suite["tiers"] if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tiers {bad} in {path}; expected {list(TIERS)}")
    return suite


def _backend(spec: VariationSpec, tier: str):
    from sim.robosuite.backend import RobosuiteBackend   # lazy: needs robosuite

    class _Backend(RobosuiteBackend):
        on_reset = None      # ExpertPolicy.begin when the expert is evaluated

        def reset_config(self, cfg):
            obs = super().reset_config(cfg)
            if self.on_reset is not None:
                self.on_reset(cfg)
            return obs

    return _Backend(spec, tier=tier)


def _policy(policy_arg: str, backend, spec: VariationSpec):
    if policy_arg == "expert":
        from engines.expert_policy import ExpertPolicy
        policy = ExpertPolicy(backend.sim, spec, control_hz=FPS)
        backend.on_reset = policy.begin
    else:
        from engines.policies import make_policy
        policy = make_policy(policy_arg, action_dim=7)
        backend.on_reset = None
    policy.load("")
    return policy


def run_episode(policy, backend, task: TaskSpec, obj: str, seed: int, max_steps: int,
                instruction: str | None = None, frames: list | None = None) -> EpisodeResult:
    instruction = instruction or task.instructions(obj)[seed % len(task.instruction_templates)]
    obs = backend.reset(None, seed=seed, target_object=obj)
    latencies: list[float] = []
    steps, info = 0, None
    while steps < max_steps:
        t0 = time.perf_counter()
        chunk = np.asarray(policy.predict(obs, instruction), dtype=np.float32).reshape(-1, 7)
        latencies.append((time.perf_counter() - t0) * 1000)
        for action in chunk:
            obs, info = backend.step(action)
            steps += 1
            if frames is not None and steps % FRAME_EVERY == 0:
                frames.append(np.concatenate([obs["agentview"], obs["eye_in_hand"]], axis=1))
            if info.success or steps >= max_steps:
                break
        if info.success:
            break
    success = bool(info is not None and info.success)
    failed = None if success else next((s for s in STAGES if not (info and info.stage_flags.get(s))), STAGES[0])
    return EpisodeResult(episode=0, seed=seed, object=obj, instruction=instruction, success=success,
                         failed_stage=failed, steps=steps,
                         latency_ms_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
                         latency_ms_p95=float(np.percentile(latencies, 95)) if latencies else 0.0)


def save_video(frames: list, stem: Path, fps: float) -> Path:
    import imageio.v2 as imageio
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg  # noqa: F401
        path = stem.with_suffix(".mp4")
        imageio.mimsave(path, frames, fps=fps)
    except ImportError:
        path = stem.with_suffix(".gif")
        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
    return path


def run_eval(suite_path, policy_arg: str, out: Path, spec_path, task_path, model_version: str | None = None,
             objects: list[str] | None = None, tiers: list[str] | None = None, max_seeds: int | None = None,
             video: str = "failures") -> EvalReport:
    if video not in VIDEO_MODES:
        raise ValueError(f"--video must be one of {VIDEO_MODES}")
    suite = load_suite(suite_path)
    spec, task = VariationSpec.load(spec_path), load_yaml(task_path, TaskSpec)
    objects = objects or suite.get("objects") or task.objects
    tiers = tiers or suite["tiers"]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tiers {bad}; expected {list(TIERS)}")
    seeds = suite["seeds"][:max_seeds] if max_seeds else suite["seeds"]
    out = Path(out)
    backend = _backend(spec, tiers[0])
    episodes: list[EpisodeResult] = []
    try:
        policy = _policy(policy_arg, backend, spec)
        for tier in tiers:
            backend.tier = tier
            for obj in objects:
                for seed in seeds:
                    frames = None if video == "none" else []
                    ep = run_episode(policy, backend, task, obj, seed, suite["max_steps"], frames=frames)
                    ep.episode, ep.scene_variant = len(episodes), tier
                    if frames and (video == "all" or not ep.success):
                        path = save_video(frames, out / "videos" / f"{tier}_{obj}_{seed}", FPS / FRAME_EVERY)
                        ep.video_path = str(path.relative_to(out))
                    episodes.append(ep)
                    verdict = "OK  " if ep.success else f"FAIL at {ep.failed_stage}"
                    print(f"[{len(episodes)}] {tier:22s} {obj:8s} seed {seed}: {verdict} ({ep.steps} steps)",
                          flush=True)
    finally:
        backend.close()
    report = EvalReport(model_version=model_version or policy_arg, dataset_versions=[], backend="robosuite",
                        episodes=episodes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(report.model_dump_json(indent=2))
    lo, hi = report.wilson_ci()
    print(f"success {report.success_rate():.0%} (95% CI {lo:.0%}-{hi:.0%}) over {len(episodes)} episodes", flush=True)
    return report


def run_rollout(policy_arg: str, out: Path, spec_path, task_path, obj: str, seed: int = 5000,
                tier: str = "in_distribution", instruction: str | None = None, max_steps: int = 400) -> dict:
    spec, task = VariationSpec.load(spec_path), load_yaml(task_path, TaskSpec)
    out = Path(out)
    backend = _backend(spec, tier)
    frames: list = []
    try:
        policy = _policy(policy_arg, backend, spec)
        ep = run_episode(policy, backend, task, obj, seed, max_steps, instruction=instruction, frames=frames)
    finally:
        backend.close()
    video = save_video(frames, out / "rollout", FPS / FRAME_EVERY) if frames else None
    result = {"success": ep.success, "steps": ep.steps, "failed_stage": ep.failed_stage,
              "latency_ms_p50": ep.latency_ms_p50, "latency_ms_p95": ep.latency_ms_p95, "object": obj,
              "tier": tier, "seed": seed, "instruction": ep.instruction, "policy": policy_arg,
              "video": video.name if video else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout.json").write_text(json.dumps(result, indent=2))
    print(f"rollout {obj} ({ep.instruction!r}): {'success' if ep.success else 'failed at ' + str(ep.failed_stage)} "
          f"in {ep.steps} steps", flush=True)
    return result
```

- [ ] **Step 4: Run the core tests and confirm they pass**

Run: `uv run pytest -q tests/engines` → all pass.

- [ ] **Step 5: Add the CLI stages.** In `engines/robosuite/cli.py`, add `import json` at the top and these functions before `STAGES`:

```python
def _dry(ctx, stage: str, **kwargs) -> bool:
    if ctx.dry_run:
        print(json.dumps({"stage": stage, **{k: str(v) for k, v in kwargs.items()}}), flush=True)
    return ctx.dry_run


def eval_(args, ctx):
    kwargs = dict(suite_path=ctx.path(args.suite), policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), model_version=args.model_version,
                  objects=_objects(args, ctx).split(",") if _objects(args, ctx) else None,
                  tiers=args.tiers.split(",") if args.tiers else None, max_seeds=args.max_seeds, video=args.video)
    if _dry(ctx, "eval", **kwargs):
        return 0
    from engines.robosuite.evaluate import run_eval
    run_eval(**kwargs)


def rollout(args, ctx):
    obj = args.object or (ctx.objects[0] if ctx.objects else "vial")
    kwargs = dict(policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), obj=obj, seed=args.seed, tier=args.tier,
                  instruction=args.instruction or None, max_steps=args.max_steps)
    if _dry(ctx, "rollout", **kwargs):
        return 0
    from engines.robosuite.evaluate import run_rollout
    result = run_rollout(**kwargs)
    return 0 if result["video"] else 1
```

Add to `STAGES`:

```python
    "eval": Stage(eval_, "closed-loop eval over suite tiers x objects x seeds -> eval_report.json + failure videos",
                  (Flag("--suite", default="configs/eval/suite_v2.yaml"),
                   Flag("--policy", default="expert", help="expert | random | url:<policy server>"),
                   Flag("--model-version", help="label in the report (default: the policy)"),
                   Flag("--tiers", help="comma-separated tiers (default: all in the suite)"),
                   Flag("--max-seeds", int, None, help="seeds per tier (default: all)"),
                   Flag("--video", default="failures", help="none | failures | all"), OBJECTS, SPEC)),
    "rollout": Stage(rollout, "one live episode -> rollout.mp4/gif + rollout.json",
                     (Flag("--policy", default="expert"), Flag("--object", help="default: first run object, else vial"),
                      Flag("--seed", int, 5000), Flag("--tier", default="in_distribution"),
                      Flag("--instruction", help="override, e.g. from the ASSIST panel"),
                      Flag("--max-steps", int, 400), SPEC)),
```

Append to `tests/engines/test_robosuite_cli.py`:

```python
def test_eval_and_rollout_dry_run(tmp_path, capsys):
    run = tmp_path / "run.yaml"
    run.write_text("selection: {objects: [vial, bottle]}\n")
    ev = argv_of(capsys, "eval", "--out", str(tmp_path), "--run", str(run), "--tiers", "in_distribution",
                 "--max-seeds", "1")
    assert ev["stage"] == "eval" and ev["policy_arg"] == "expert" and ev["objects"] == "['vial', 'bottle']"
    assert ev["tiers"] == "['in_distribution']" and ev["max_seeds"] == "1"
    assert ev["suite_path"] == str(ROOT / "configs/eval/suite_v2.yaml")
    ro = argv_of(capsys, "rollout", "--out", str(tmp_path), "--run", str(run), "--instruction", "Pass the vial")
    assert ro["obj"] == "vial" and ro["seed"] == "5000" and ro["instruction"] == "Pass the vial"
```

Set `eval: ready` and `rollout: ready` in `engines/robosuite/engine.yaml`, and add `eval` and `rollout` to the loop in `test_repo_engines_load_and_robosuite_is_ready_for_data`.

- [ ] **Step 6: robosuite-env tests.** Create `tests/engines/test_robosuite_eval.py`:

```python
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest  # noqa: E402

pytest.importorskip("robosuite")

from engines.registry import ROOT  # noqa: E402
from engines.robosuite.evaluate import run_eval, run_rollout  # noqa: E402

SPEC = ROOT / "configs/variation/cathlab_v1.yaml"
TASK = ROOT / "configs/tasks/cathlab_pick.yaml"
SUITE = ROOT / "configs/eval/suite_v2.yaml"


def test_expert_eval_is_a_strong_reference(tmp_path):
    report = run_eval(SUITE, "expert", tmp_path, SPEC, TASK, tiers=["in_distribution"], max_seeds=1, video="failures")
    assert len(report.episodes) == 3 and report.success_rate() >= 2 / 3
    assert {e.scene_variant for e in report.episodes} == {"in_distribution"}
    saved = json.loads((tmp_path / "eval_report.json").read_text())
    assert saved["backend"] == "robosuite" and len(saved["episodes"]) == 3


def test_random_rollout_writes_video_and_json(tmp_path):
    result = run_rollout("random", tmp_path, SPEC, TASK, "vial", max_steps=20)
    assert result["success"] is False and result["steps"] == 20
    assert (tmp_path / result["video"]).exists()
    assert json.loads((tmp_path / "rollout.json").read_text())["object"] == "vial"
```

Create `tests/engines/test_conformance.py`:

```python
"""Engine conformance: every engine whose backend imports in this env must follow the engine contract.

    MUJOCO_GL=egl conda run -n robosuite python -m pytest -q tests/engines/test_conformance.py -k robosuite

In the core env the simulators are missing, so these tests skip.
"""
import importlib
import json
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from contracts import STAGES, StepInfo, TaskSpec, load_yaml  # noqa: E402
from engines.registry import ROOT, load_engines  # noqa: E402

ENGINES = [m for m in load_engines().values() if m.backend]


def backend_cls(m):
    module, _, cls = m.backend.partition(":")
    try:
        return getattr(importlib.import_module(module), cls)
    except ImportError as e:
        pytest.skip(f"{m.id} backend not importable in this env: {e}")


@pytest.mark.parametrize("m", ENGINES, ids=lambda m: m.id)
def test_backend_follows_io_profile(m):
    backend = backend_cls(m)()
    target = load_yaml(ROOT / m.task, TaskSpec).objects[0]
    try:
        obs = backend.reset(None, seed=5000, target_object=target)
        for key, shape in m.io_profile.obs_shapes.items():
            assert obs[key].shape == shape, key
        obs, info = backend.step(np.zeros(m.io_profile.action_dim, dtype=np.float32))
        assert isinstance(info, StepInfo) and set(info.stage_flags) == set(STAGES) and not info.success
        assert backend.render().ndim == 3
    finally:
        backend.close()


@pytest.mark.parametrize("m", ENGINES, ids=lambda m: m.id)
def test_expert_reaches_80_percent(m, tmp_path):
    backend_cls(m)
    out = tmp_path / "check_expert"
    rc = subprocess.run([sys.executable, "-m", m.entry, "check-expert", "--out", str(out), "--n", "5"],
                        cwd=ROOT).returncode
    assert rc == 0
    for obj, v in json.loads((out / "expert_check.json").read_text())["per_object"].items():
        assert v["rate"] >= 0.8, f"{obj}: {v}"
```

Run:

```bash
uv run pytest -q tests/engines        # core: conformance + eval tests skip
MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m pytest -q tests/engines
```

Expected: core passes. robosuite env: `test_robosuite_eval.py` passes (2) and `test_conformance.py` passes for `robosuite` (2). If the expert reference is below 2/3 on `in_distribution` seed 5000, report it as DONE_WITH_CONCERNS with the per-episode log: don't lower the bar.

- [ ] **Step 7: Real CLI smoke** (robosuite env):

```bash
MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m engines.robosuite.cli rollout \
    --out var/smoke/rollout --policy expert --object vial
cat var/smoke/rollout/rollout.json      # "success": true, "video": "rollout.mp4" or "rollout.gif"
```

- [ ] **Step 8: Commit**

```bash
git add engines configs/eval/suite_v2.yaml tests/engines
git commit -m "feat(engines): robosuite eval and rollout stages, expert as policy, suite v2, engine conformance test"
```

---

### Task 5: Stage pipeline: stage registry, worker dispatch, job logs and outputs

**Files:**
- Create: `orchestrator/stages.py`, `orchestrator/pipeline.py`, `tests/engines/fake/engine.yaml`, `tests/engines/fake_engine_cli.py`, `tests/orchestrator/test_pipeline.py`
- Modify: `orchestrator/control.py`

**Interfaces:**
- Consumes:
  - `engines.registry.load_engines`, `EngineManifest.status`, `.worker`, `.entry` (Task 1); `engines.cli_base` (Task 2);
  - `orchestrator.store.Store`:
    - `create_job(kind, name, params, run_id, worker, state="queued") -> Job`;
    - `update_job(job_id, state=None, result=None, worker=None) -> Job` (the result is merged);
    - `list_jobs(kind, run_id)`, `get_job`, `get_run`.
  - Worker API:
    - `POST /jobs {argv, name, cwd, env, timeout_s}` → `{id, state, returncode, …}`;
    - `GET /jobs/{id}`;
    - `GET /jobs/{id}/log?offset` → `{text, offset, state}`;
    - `POST /jobs/{id}/cancel`;
    - `GET /health` → `{name, capabilities, executable, …}`.
    - Worker job states: `running | done | failed | cancelled | timeout`.
- Produces:
  - `orchestrator.stages`: `StageParam(name, type="str", default=None, help="")`, `StageDef(id, label, cli, capability, worker_capability=None, needs=(), inputs={}, params=())`, `STAGES`, `STAGE_BY_ID`, `stage_argv(stage, params) -> list[str]`.
  - `orchestrator.pipeline`:
    - `PipelineError(message, status=409)`, `worker_headers()`, `worker_request(method, url, timeout=5.0, **kw) -> httpx.Response`;
    - `Pipeline(store, var_dir, repo_root, workers_fn, engines_fn, http=worker_request)` with methods:
      - `.workers()` → health dicts;
      - `.stages(run) -> list[dict]`;
      - `.start(run, stage_id, params) -> Job`;
      - `.refresh(job) -> Job`;
      - `.log(job, offset) -> dict`;
      - `.cancel(job) -> Job`;
      - `.files(job) -> list[dict]`;
      - `.file(job, rel) -> Path`.
  - `create_control_app(..., engines_dir=None, worker_http=None)`.
  - HTTP:
    - `GET /api/engines`;
    - `GET /api/runs/{id}/stages`;
    - `POST /api/runs/{id}/stages/{stage}` with body `{params: {}}`;
    - `GET /api/jobs/{id}/log?offset=`;
    - `POST /api/jobs/{id}/cancel`;
    - `GET /api/jobs/{id}/files`;
    - `GET /api/jobs/{id}/file?path=`.
  - Stage dict shape (used by Task 6): `{id, label, params: [{name, type, default, help}], enabled: bool, reason: str | None, worker: str | None, job: Job dict | None}`.
  - A stage job's `result`: `{worker_job, worker_url, argv, out, returncode?, error?}`. Outputs go to `var/runs/<run>/<stage>/<job id>/`.

- [ ] **Step 1: Create the fake engine.** `tests/engines/fake/engine.yaml`:

```yaml
id: fake
label: Fake engine for pipeline tests
env: {kind: conda, name: core}
worker: fake
embodiments: [panda]
io: panda_libero
variation: configs/variation/cathlab_v1.yaml
task: configs/tasks/cathlab_pick.yaml
capabilities:
  preview: ready
  expert: {analytic: ready}
  generate: ready
  eval: planned
entry: tests.engines.fake_engine_cli
```

`tests/engines/fake_engine_cli.py`:

```python
"""A tiny engine for pipeline tests: its stages write small files into --out."""
from pathlib import Path
from engines.cli_base import Flag, Stage, run_cli
from engines.registry import load_manifest

MANIFEST = load_manifest(Path(__file__).parent / "fake" / "engine.yaml")


def preview(args, ctx):
    print(f"preview n={args.n} objects={','.join(ctx.objects)}", flush=True)
    (ctx.out / "grid.txt").write_text(f"n={args.n}")


def generate(args, ctx):
    (ctx.out / "raw").mkdir()
    (ctx.out / "raw" / "ok").write_text("1")


def convert(args, ctx):
    (ctx.out / "lerobot.txt").write_text(args.dataset)


STAGES = {
    "preview": Stage(preview, flags=(Flag("--n", int, 8),)),
    "check-expert": Stage(lambda a, c: None, flags=(Flag("--n", int, 5), Flag("--split", default="train"))),
    "generate": Stage(generate, flags=(Flag("--n", int, 30), Flag("--start", int, 0), Flag("--split", default="train"))),
    "convert": Stage(convert, flags=(Flag("--dataset", required=True),)),
}

if __name__ == "__main__":
    raise SystemExit(run_cli("fake", STAGES, manifest=MANIFEST))
```

- [ ] **Step 2: Write the failing tests** in `tests/orchestrator/test_pipeline.py`:

```python
import time
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from orchestrator.control import create_control_app
from orchestrator.stages import STAGE_BY_ID, stage_argv
from orchestrator.worker import create_worker_app

ROOT = Path(__file__).resolve().parents[2]
CATALOG = """catalog_version: 1
sections:
  simulator: {title: Simulator, items: [{id: fake, label: Fake, status: ready}]}
  expert: {title: Expert, items: [{id: analytic, label: Analytic, status: ready}]}
  objects: {title: Objects, multi: true, items: [{id: vial, label: Vial, status: ready}]}
defaults: {simulator: fake, expert: analytic, objects: [vial]}
"""


def router(clients: dict):
    def http(method, url, timeout=None, **kw):
        for base, client in clients.items():
            if url.startswith(base):
                return client.request(method, url[len(base):], **kw)
        raise httpx.ConnectError(f"no route to {url}")
    return http


def control(tmp_path, worker_online=True):
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    (tmp_path / "workers.yaml").write_text('workers:\n  - {name: fake, url: "http://fake-worker"}\n')
    clients = {}
    if worker_online:
        clients["http://fake-worker"] = TestClient(create_worker_app("fake", ["convert"], tmp_path / "worker"))
    app = create_control_app(ROOT, var_dir=tmp_path / "var", catalog_path=tmp_path / "catalog.yaml",
                             workers_path=tmp_path / "workers.yaml", frontend_dir=tmp_path / "none",
                             engines_dir=ROOT / "tests/engines", worker_http=router(clients))
    return TestClient(app)


def wait_done(c, job_id, timeout=30):
    end, state, text, offset = time.time() + timeout, "", "", 0
    while time.time() < end:
        r = c.get(f"/api/jobs/{job_id}/log", params={"offset": offset}).json()
        text, offset, state = text + r["text"], r["offset"], r["state"]
        if state not in ("queued", "running"):
            break
        time.sleep(0.1)
    return state, text


def stages(c, run_id):
    return {s["id"]: s for s in c.get(f"/api/runs/{run_id}/stages").json()}


def test_stage_argv_skips_empty_and_validates():
    gen = STAGE_BY_ID["generate"]
    assert stage_argv(gen, {"n": "6", "start": "", "split": "train"}) == ["--n", "6", "--split", "train"]
    with pytest.raises(ValueError, match="no parameter 'bogus'"):
        stage_argv(gen, {"bogus": 1})
    with pytest.raises(ValueError):
        stage_argv(gen, {"n": "six"})


def test_pipeline_end_to_end_with_fake_engine(tmp_path):
    c = control(tmp_path)
    assert [e["id"] for e in c.get("/api/engines").json()] == ["fake"]
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    st = stages(c, run["id"])
    assert st["preview"]["enabled"] and st["preview"]["worker"] == "fake"
    assert st["convert"]["reason"] == "needs 'generate' to finish first"
    assert st["eval"]["reason"] == "fake: 'eval' is planned" and not st["eval"]["enabled"]

    job = c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {"n": 3}}).json()
    state, text = wait_done(c, job["id"])
    assert state == "done" and "preview n=3 objects=vial" in text
    assert [f["path"] for f in c.get(f"/api/jobs/{job['id']}/files").json()] == ["grid.txt"]
    assert c.get(f"/api/jobs/{job['id']}/file", params={"path": "grid.txt"}).text == "n=3"
    assert c.get(f"/api/jobs/{job['id']}/file", params={"path": "../../run.yaml"}).status_code == 400

    gen = c.post(f"/api/runs/{run['id']}/stages/generate", json={"params": {"n": 2}}).json()
    assert wait_done(c, gen["id"])[0] == "done"
    assert stages(c, run["id"])["convert"]["enabled"]
    conv = c.post(f"/api/runs/{run['id']}/stages/convert", json={"params": {}}).json()
    assert wait_done(c, conv["id"])[0] == "done"
    lerobot = c.get(f"/api/jobs/{conv['id']}/file", params={"path": "lerobot.txt"}).text
    assert lerobot == c.get(f"/api/jobs/{gen['id']}").json()["result"]["out"]

    assert c.post(f"/api/runs/{run['id']}/stages/eval", json={"params": {}}).status_code == 409
    assert c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {"bogus": 1}}).status_code == 400
    assert c.post(f"/api/runs/{run['id']}/stages/nope", json={"params": {}}).status_code == 404


def test_offline_worker_blocks_stages(tmp_path):
    c = control(tmp_path, worker_online=False)
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    st = stages(c, run["id"])
    assert st["preview"]["reason"] == "worker 'fake' is offline"
    assert st["convert"]["reason"] == "no online worker with capability 'convert'"
    assert c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {}}).status_code == 409


def test_cancel_running_stage(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    job = c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {}}).json()
    assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    assert wait_done(c, job["id"])[0] in ("cancelled", "done")
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `uv run pytest -q tests/orchestrator/test_pipeline.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestrator.stages'`.

- [ ] **Step 4: Implement** `orchestrator/stages.py`:

```python
"""Pipeline stages of a run: which engine CLI subcommand, on which worker, after what, with which params."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageParam:
    name: str                 # engine CLI flag without "--", e.g. "n", "max-seeds"
    type: str = "str"         # "str" | "int"
    default: Any = None
    help: str = ""


@dataclass(frozen=True)
class StageDef:
    id: str
    label: str
    cli: str                                   # engine CLI subcommand
    capability: str                            # manifest capability; "{expert}" = the run's expert id
    worker_capability: str | None = None       # None: the engine's worker; else any healthy worker advertising it
    needs: tuple = ()                          # stages whose latest job must be `done`
    inputs: dict = field(default_factory=dict)  # CLI flag -> stage whose output folder it receives
    params: tuple = ()

    def capability_for(self, selection: dict) -> str:
        return self.capability.format(expert=selection.get("expert") or "analytic")

    def param_dicts(self) -> list[dict]:
        return [asdict(p) for p in self.params]


STAGES: tuple = (
    StageDef("preview", "Scene preview", "preview", "preview",
             params=(StageParam("n", "int", 8, "train configs in the grid"),)),
    StageDef("check_expert", "Check expert", "check-expert", "expert.{expert}",
             params=(StageParam("n", "int", 5, "episodes per object"), StageParam("split", "str", "train"))),
    StageDef("generate", "Generate data", "generate", "generate",
             params=(StageParam("n", "int", 30, "configs (all objects, balanced)"),
                     StageParam("start", "int", 0, "first config of the window"), StageParam("split", "str", "train"))),
    StageDef("convert", "Convert to LeRobot", "convert", "generate", worker_capability="convert",
             needs=("generate",), inputs={"dataset": "generate"}),
    StageDef("eval", "Evaluate", "eval", "eval",
             params=(StageParam("policy", "str", "expert", "expert | random | url:<policy server>"),
                     StageParam("model-version", "str", "", "label in the report"),
                     StageParam("tiers", "str", "in_distribution", "comma-separated; empty = all suite tiers"),
                     StageParam("max-seeds", "int", 1, "seeds per tier; empty = all"))),
    StageDef("rollout", "Rollout", "rollout", "rollout",
             params=(StageParam("policy", "str", "expert", "expert | random | url:<policy server>"),
                     StageParam("object", "str", "", "default: first object of the run"),
                     StageParam("seed", "int", 5000), StageParam("instruction", "str", "", "override"))),
)
STAGE_BY_ID = {s.id: s for s in STAGES}


def stage_argv(stage: StageDef, params: dict) -> list[str]:
    """CLI flags for the user's params. Empty values are left out, so the engine's defaults apply."""
    known = {p.name: p for p in stage.params}
    argv: list[str] = []
    for name, value in params.items():
        if name not in known:
            raise ValueError(f"stage '{stage.id}' has no parameter '{name}'")
        if value is None or value == "":
            continue
        if known[name].type == "int":
            value = int(value)
        argv += [f"--{name}", str(value)]
    return argv
```

`orchestrator/pipeline.py`:

```python
"""Run pipeline: which stages of a run can start, dispatch to workers, follow jobs, list their outputs."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable
import httpx
from orchestrator.stages import STAGE_BY_ID, STAGES, StageDef, stage_argv
from orchestrator.store import Job, Run, Store

ACTIVE = ("queued", "running")
MAX_FILES = 200


class PipelineError(Exception):
    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def worker_headers() -> dict[str, str]:
    token = os.environ.get("CATHLAB_WORKER_TOKEN")
    return {"X-Cathlab-Token": token} if token else {}


def worker_request(method: str, url: str, timeout: float = 5.0, **kw) -> httpx.Response:
    return httpx.request(method, url, headers=worker_headers(), timeout=timeout, **kw)


class Pipeline:
    def __init__(self, store: Store, var_dir: Path, repo_root: Path, workers_fn: Callable[[], list],
                 engines_fn: Callable[[], dict], http: Callable[..., httpx.Response] = worker_request):
        self.store, self.var_dir, self.repo_root = store, Path(var_dir), Path(repo_root)
        self._workers_fn, self._engines_fn, self._http = workers_fn, engines_fn, http

    # workers ---------------------------------------------------------------------------------
    def health(self, entry) -> dict:
        try:
            r = self._http("GET", entry.url.rstrip("/") + "/health", timeout=0.5)
            r.raise_for_status()
            return {"name": entry.name, "url": entry.url, "healthy": True, **r.json(), "error": ""}
        except Exception as e:  # noqa: BLE001 - any failure means "offline"
            return {"name": entry.name, "url": entry.url, "healthy": False, "capabilities": [], "error": str(e)[:200]}

    def workers(self) -> list[dict]:
        entries = self._workers_fn()
        with ThreadPoolExecutor(max_workers=max(1, len(entries))) as pool:
            return list(pool.map(self.health, entries))

    # stages ----------------------------------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.var_dir / "runs" / run_id

    def latest(self, run_id: str, stage_id: str) -> Job | None:
        jobs = [j for j in self.store.list_jobs(kind="stage", run_id=run_id) if j.name == stage_id]
        return max(jobs, key=lambda j: j.created, default=None)

    def _check(self, run: Run, stage: StageDef, engines: dict, workers: list) -> tuple:
        """(reason the stage can't start or None, the worker it would run on or None)."""
        engine_id = run.selection.get("simulator")
        manifest = engines.get(engine_id)
        if manifest is None:
            return f"no engine plugin '{engine_id}' (engines/{engine_id}/engine.yaml)", None
        cap = stage.capability_for(run.selection)
        status = manifest.status(cap)
        if status != "ready":
            return f"{engine_id}: '{cap}' is {status.replace('_', ' ')}", None
        if stage.worker_capability:
            worker = next((w for w in workers if w["healthy"] and stage.worker_capability in w.get("capabilities", [])),
                          None)
            if worker is None:
                return f"no online worker with capability '{stage.worker_capability}'", None
        else:
            worker = next((w for w in workers if w["name"] == manifest.worker), None)
            if worker is None or not worker["healthy"]:
                return f"worker '{manifest.worker}' is offline", None
        for need in stage.needs:
            job = self.latest(run.id, need)
            if job is None or job.state != "done":
                return f"needs '{need}' to finish first", worker
        return None, worker

    def stages(self, run: Run) -> list[dict]:
        engines, workers = self._engines_fn(), self.workers()
        out = []
        for stage in STAGES:
            reason, worker = self._check(run, stage, engines, workers)
            job = self.latest(run.id, stage.id)
            if job is not None and job.state in ACTIVE:
                job = self.refresh(job)
            running = job is not None and job.state in ACTIVE
            out.append({"id": stage.id, "label": stage.label, "params": stage.param_dicts(),
                        "enabled": reason is None and not running, "reason": reason,
                        "worker": worker["name"] if worker else None, "job": job.model_dump() if job else None})
        return out

    def start(self, run: Run, stage_id: str, params: dict) -> Job:
        stage = STAGE_BY_ID.get(stage_id)
        if stage is None:
            raise PipelineError(f"unknown stage '{stage_id}'", 404)
        engines = self._engines_fn()
        reason, worker = self._check(run, stage, engines, self.workers())
        if reason:
            raise PipelineError(reason)
        latest = self.latest(run.id, stage_id)
        if latest is not None and latest.state in ACTIVE:
            raise PipelineError(f"'{stage_id}' is already running ({latest.id})")
        try:
            flags = stage_argv(stage, params)
        except ValueError as e:
            raise PipelineError(str(e), 400) from e
        manifest = engines[run.selection["simulator"]]
        job = self.store.create_job("stage", stage.id, params, run_id=run.id, worker=worker["name"])
        rd = self.run_dir(run.id)
        out = rd / stage.id / job.id
        argv = [worker.get("executable") or "python", "-m", manifest.entry, stage.cli,
                "--run", str(rd / "run.yaml"), "--out", str(out), *flags]
        for flag, source in stage.inputs.items():
            argv += [f"--{flag}", self.latest(run.id, source).result["out"]]
        try:
            r = self._http("POST", worker["url"].rstrip("/") + "/jobs",
                           json={"argv": argv, "name": f"{run.id}:{stage.id}", "cwd": str(self.repo_root)})
            r.raise_for_status()
        except Exception as e:  # noqa: BLE001
            self.store.update_job(job.id, state="failed",
                                  result={"error": f"dispatch failed: {e}"[:300], "argv": argv, "out": str(out)})
            raise PipelineError(f"could not start on worker '{worker['name']}': {e}", 502) from e
        return self.store.update_job(job.id, state="running", result={
            "worker_job": r.json()["id"], "worker_url": worker["url"], "argv": argv, "out": str(out)})

    # jobs ------------------------------------------------------------------------------------
    def _worker_url(self, job: Job, suffix: str = "") -> str:
        return f"{job.result['worker_url'].rstrip('/')}/jobs/{job.result['worker_job']}{suffix}"

    def refresh(self, job: Job) -> Job:
        if job.state not in ACTIVE or "worker_job" not in job.result:
            return job
        try:
            r = self._http("GET", self._worker_url(job))
        except Exception:  # noqa: BLE001 - worker unreachable: keep the last known state
            return job
        if r.status_code == 404:
            return self.store.update_job(job.id, state="failed", result={"error": "the worker lost the job (restarted?)"})
        if r.status_code != 200:
            return job
        state = r.json()
        if state["state"] != job.state:
            job = self.store.update_job(job.id, state=state["state"], result={"returncode": state.get("returncode")})
        return job

    def log(self, job: Job, offset: int = 0) -> dict:
        job = self.refresh(job)
        if "worker_job" not in job.result:
            return {"text": job.result.get("error", ""), "offset": offset, "state": job.state}
        try:
            r = self._http("GET", self._worker_url(job, "/log"), params={"offset": offset})
            r.raise_for_status()
            data = r.json()
            return {"text": data["text"], "offset": data["offset"], "state": job.state}
        except Exception as e:  # noqa: BLE001
            return {"text": "", "offset": offset, "state": job.state, "error": str(e)[:200]}

    def cancel(self, job: Job) -> Job:
        if job.state in ACTIVE and "worker_job" in job.result:
            try:
                self._http("POST", self._worker_url(job, "/cancel"))
            except Exception as e:  # noqa: BLE001
                raise PipelineError(f"cancel failed: {e}", 502) from e
        return self.refresh(job)

    def _out(self, job: Job) -> Path:
        if "out" not in job.result:
            raise PipelineError("job has no outputs", 404)
        return Path(job.result["out"]).resolve()

    def files(self, job: Job) -> list[dict]:
        base = self._out(job)
        if not base.exists():
            return []
        files = sorted(p for p in base.rglob("*") if p.is_file())[:MAX_FILES]
        return [{"path": str(p.relative_to(base)), "size": p.stat().st_size} for p in files]

    def file(self, job: Job, rel: str) -> Path:
        base = self._out(job)
        path = (base / rel).resolve()
        if base not in path.parents:
            raise PipelineError("path is outside the job's outputs", 400)
        if not path.is_file():
            raise PipelineError(f"no file '{rel}'", 404)
        return path
```

- [ ] **Step 5: Wire it into `orchestrator/control.py`**:
  - Remove `worker_health`; workers now come from `Pipeline.workers()`.
  - Add the imports:
    - `from fastapi.responses import FileResponse`;
    - `from engines.registry import ENGINES_DIR, load_engines`;
    - `from orchestrator.pipeline import Pipeline, PipelineError, worker_request`.
  - Add `class StageStart(BaseModel): params: dict = Field(default_factory=dict)`.
  - Add the keyword parameters `engines_dir: str | Path | None = None, worker_http=None` to `create_control_app`, and after `app.state.store = store` add:

```python
    engines_dir = Path(engines_dir or ENGINES_DIR)
    pipeline = Pipeline(store, var_dir, repo_root, workers_fn=lambda: load_workers(workers_path).workers,
                        engines_fn=lambda: load_engines(engines_dir), http=worker_http or worker_request)
    app.state.pipeline = pipeline

    def run_or_404(run_id: str):
        r = store.get_run(run_id)
        if r is None:
            raise HTTPException(404, f"unknown run {run_id}")
        return r

    def job_or_404(job_id: str):
        j = store.get_job(job_id)
        if j is None:
            raise HTTPException(404, f"unknown job {job_id}")
        return j

    def pipeline_call(fn, *args):
        try:
            return fn(*args)
        except PipelineError as e:
            raise HTTPException(e.status, str(e)) from e
```

Replace the bodies of the existing `run`, `job` and `workers` endpoints, and add the new ones. Keep their route paths:

```python
    @app.get("/api/runs/{run_id}")
    def run(run_id: str) -> dict:
        return run_or_404(run_id).model_dump()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict:
        return pipeline.refresh(job_or_404(job_id)).model_dump()

    @app.get("/api/workers")
    def workers() -> list[dict]:
        return pipeline.workers()

    @app.get("/api/engines")
    def engines() -> list[dict]:
        return [m.model_dump() for m in load_engines(engines_dir).values()]

    @app.get("/api/runs/{run_id}/stages")
    def run_stages(run_id: str) -> list[dict]:
        return pipeline.stages(run_or_404(run_id))

    @app.post("/api/runs/{run_id}/stages/{stage_id}")
    def start_stage(run_id: str, stage_id: str, body: StageStart) -> dict:
        return pipeline_call(pipeline.start, run_or_404(run_id), stage_id, body.params).model_dump()

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, offset: int = 0) -> dict:
        return pipeline.log(job_or_404(job_id), offset)

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> dict:
        return pipeline_call(pipeline.cancel, job_or_404(job_id)).model_dump()

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str) -> list[dict]:
        return pipeline_call(pipeline.files, job_or_404(job_id))

    @app.get("/api/jobs/{job_id}/file")
    def job_file(job_id: str, path: str):
        return FileResponse(pipeline_call(pipeline.file, job_or_404(job_id), path))
```

Keep `load_workers`, `WorkerEntry` and `WorkersConfig` in `control.py`; `Pipeline.health` reads `entry.name` and `entry.url`. Also check `scripts/demo_control_station.py` still runs: it calls `create_control_app` with keyword arguments that remain valid.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator` → all pass. This includes the existing `test_control.py` worker-health test, which goes through `pipeline.workers()` against a real uvicorn worker. Then `uv run pytest -q` → all pass.

- [ ] **Step 7: Real smoke**:
  - Run `make dev` and `make worker-robosuite` in two terminals.
  - Create a run with the defaults:

```bash
RUN=$(curl -s -X POST localhost:8700/api/runs -H 'Content-Type: application/json' \
  -d "$(curl -s localhost:8700/api/catalog | python -c 'import json,sys; print(json.dumps({"selection": json.load(sys.stdin)["defaults"]}))')" \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
curl -s localhost:8700/api/runs/$RUN/stages | python -m json.tool | grep -E '"id"|"reason"'
JOB=$(curl -s -X POST localhost:8700/api/runs/$RUN/stages/preview -H 'Content-Type: application/json' \
  -d '{"params": {"n": 2}}' | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
sleep 60; curl -s "localhost:8700/api/jobs/$JOB/files"
```

  - Expected:
    - preview, check_expert, generate, eval and rollout are enabled;
    - convert says `no online worker with capability 'convert'` (no lerobot worker running);
    - after the preview job finishes, `files` lists `grid.png`.
  - Stop both servers afterwards (find the PID with `ss -ltnp | grep -E ':87(00|11)'` and `kill <PID>`).

- [ ] **Step 8: Commit**

```bash
git add orchestrator tests/orchestrator/test_pipeline.py tests/engines/fake tests/engines/fake_engine_cli.py
git commit -m "feat(orchestrator): stage pipeline dispatching engine CLIs to workers, job logs and outputs API"
```

---

### Task 6: PIPELINE tab: run selector, stage strip, params, live log, outputs

**Files:**
- Create: `frontend/js/pipeline.js`
- Modify: `frontend/index.html` (the PIPELINE section and footer), `frontend/js/app.js` (remove the old strip, call `initPipeline`, refresh runs after CREATE RUN), `frontend/css/station.css`

**Interfaces:**
- Consumes the Task 5 HTTP API and stage dict shape; `api`, `$`, `el`, `setStatus` from `api.js`/`dom.js`; `showTab` from `app.js`.
- Produces:
  - `export async function initPipeline()`, `export async function refreshRuns(selectId)`;
  - `export function stageGate(stageId)`: stage → gate id, used by Task 8 (`preview→scene`, `check_expert→expert`, `generate→data_method`, `convert→convert`, `eval→eval`, `rollout→rollout`).

- [ ] **Step 1: Markup.** In `frontend/index.html`, replace the PIPELINE section with:

```html
    <!-- PIPELINE: run selector + stage strip + stage detail -->
    <section class="tab hidden" id="tab-pipeline">
      <div class="row runbar">
        <label class="panel-title" for="run-select">RUN</label>
        <select id="run-select" class="field"></select>
      </div>
      <div class="strip" id="strip"></div>
      <div class="pane">
        <div class="panel-title" id="stage-title">STAGE</div>
        <div id="stage-detail" class="muted">Select a stage.</div>
      </div>
    </section>
```

In the footer, change the RUN button to `<button class="btn run" id="btn-run" disabled title="Run the selected stage">&#9654; RUN STAGE</button>`.

- [ ] **Step 2: Create `frontend/js/pipeline.js`:**

```javascript
// PIPELINE tab: pick a run, see which stages can start (and why not), run one, follow its log and outputs.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";

const ps = { runId: null, stages: [], current: null, jobId: null, offset: 0, timer: null };
const GATES = { preview: "scene", check_expert: "expert", generate: "data_method", convert: "convert",
  eval: "eval", rollout: "rollout" };
const ACTIVE = ["queued", "running"];

export function stageGate(stageId) { return GATES[stageId] || null; }

export async function initPipeline() {
  $("#run-select").addEventListener("change", (e) => selectRun(e.target.value));
  $("#btn-run").addEventListener("click", runCurrent);
  $("#btn-stop").addEventListener("click", stopCurrent);
  await refreshRuns();
  setInterval(refreshStages, 3000);
}

export async function refreshRuns(selectId) {
  const runs = await api.get("/api/runs");
  const sel = $("#run-select");
  sel.innerHTML = "";
  if (!runs.length) sel.append(new Option("No runs yet: CREATE RUN in PROJECT", ""));
  for (const r of runs) sel.append(new Option(`${r.id}  ·  ${r.selection.simulator}  ${r.note || ""}`, r.id));
  const id = selectId || ps.runId || (runs[0] ? runs[0].id : "");
  sel.value = id;
  await selectRun(id);
}

async function selectRun(id) {
  ps.runId = id || null;
  ps.current = null;
  stopFollow();
  await refreshStages();
  renderDetail();
}

async function refreshStages() {
  if (!ps.runId) { ps.stages = []; renderStrip(); updateFooter(); return; }
  try {
    ps.stages = await api.get(`/api/runs/${ps.runId}/stages`);
  } catch (e) { setStatus(`STAGES: ${e.message}`); return; }
  renderStrip();
  updateFooter();
}

function current() { return ps.stages.find((s) => s.id === ps.current) || null; }

function lamp(s) {
  const st = s.job ? s.job.state : null;
  if (ACTIVE.includes(st)) return "running";
  if (st === "done") return "done";
  if (st) return "failed";
  return s.enabled ? "ready" : "blocked";
}

function renderStrip() {
  const strip = $("#strip");
  strip.innerHTML = "";
  for (const s of ps.stages) {
    const t = el("div", `stage ${lamp(s)}${ps.current === s.id ? " active" : ""}`);
    t.append(el("div", "", s.label.toUpperCase()),
      el("div", "sub", s.job ? s.job.state : (s.enabled ? "ready" : "blocked")));
    t.title = s.reason || "";
    t.addEventListener("click", () => { ps.current = s.id; renderStrip(); renderDetail(); });
    strip.append(t);
  }
}

function renderDetail() {
  const s = current();
  const box = $("#stage-detail");
  box.innerHTML = "";
  $("#stage-title").textContent = s ? s.label.toUpperCase() : "STAGE";
  if (!s) {
    box.append(el("div", "muted", ps.runId ? "Select a stage." : "Create a run in PROJECT first."));
    updateFooter();
    return;
  }
  box.append(el("div", s.reason ? "p" : "ok", s.reason ? `BLOCKED: ${s.reason}` : `Runs on worker ${s.worker}.`));
  const form = el("form", "form");
  form.id = "stage-form";
  for (const p of s.params) {
    const input = el("input", "field");
    input.name = p.name;
    input.value = p.default ?? "";
    if (p.type === "int") input.type = "number";
    form.append(el("label", "", p.name), input);
    if (p.help) form.append(el("div", "help", p.help));
  }
  form.addEventListener("submit", (e) => { e.preventDefault(); runCurrent(); });
  const meta = el("div", "kv mono", "no job yet");
  meta.id = "stage-job";
  const con = el("pre", "console");
  con.id = "stage-console";
  const files = el("div", "files");
  files.id = "stage-files";
  box.append(form, meta, con, files);
  if (s.job) followJob(s.job.id); else stopFollow();
  updateFooter();
}

function updateFooter() {
  const s = current();
  const running = !!(s && s.job && ACTIVE.includes(s.job.state));
  $("#btn-run").disabled = !(s && s.enabled);
  $("#btn-stop").disabled = !running;
}

function stopFollow() {
  if (ps.timer) clearInterval(ps.timer);
  ps.timer = null;
  ps.jobId = null;
}

function followJob(id) {
  stopFollow();
  ps.jobId = id;
  ps.offset = 0;
  $("#stage-console").textContent = "";
  $("#stage-files").innerHTML = "";
  const tick = async () => {
    if (ps.jobId !== id) return;
    try {
      const r = await api.get(`/api/jobs/${id}/log?offset=${ps.offset}`);
      if (ps.jobId !== id) return;
      const con = $("#stage-console");
      if (r.text) { con.textContent += r.text; con.scrollTop = con.scrollHeight; }
      ps.offset = r.offset;
      $("#stage-job").textContent = `JOB ${id} · ${r.state.toUpperCase()}`;
      if (!ACTIVE.includes(r.state)) {
        if (ps.timer) clearInterval(ps.timer);
        ps.timer = null;
        showFiles(id);
        refreshStages();
      }
    } catch (e) { $("#stage-job").textContent = `JOB ${id} · ${e.message}`; }
  };
  ps.timer = setInterval(tick, 1000);
  tick();
}

async function showFiles(id) {
  const box = $("#stage-files");
  box.innerHTML = "";
  const files = await api.get(`/api/jobs/${id}/files`).catch(() => []);
  if (!files.length) return;
  box.append(el("div", "panel-title", "OUTPUTS"));
  let media = 0;
  for (const f of files) {
    const url = `/api/jobs/${id}/file?path=${encodeURIComponent(f.path)}`;
    const a = el("a", "mono", `${f.path}  (${(f.size / 1024).toFixed(1)} kB)`);
    a.href = url;
    a.target = "_blank";
    box.append(a);
    if (media < 4 && /\.(png|gif|jpg)$/i.test(f.path)) { const img = el("img"); img.src = url; box.append(img); media++; }
    if (media < 4 && /\.mp4$/i.test(f.path)) { const v = el("video"); v.src = url; v.controls = true; box.append(v); media++; }
  }
}

async function runCurrent() {
  const s = current();
  if (!s || !s.enabled) return;
  const params = Object.fromEntries(new FormData($("#stage-form")).entries());
  try {
    const job = await api.post(`/api/runs/${ps.runId}/stages/${s.id}`, { params });
    setStatus(`STARTED ${s.label.toUpperCase()} (${job.id})`);
    await refreshStages();
    followJob(job.id);
  } catch (e) { setStatus(`STAGE REJECTED: ${e.message}`); }
}

async function stopCurrent() {
  const s = current();
  if (!s || !s.job) return;
  try {
    await api.post(`/api/jobs/${s.job.id}/cancel`);
    setStatus(`STOP SENT (${s.job.id})`);
  } catch (e) { setStatus(`STOP FAILED: ${e.message}`); }
  refreshStages();
}
```

- [ ] **Step 3: Wire it into `frontend/js/app.js`:**
  - Delete the old `STAGES` constant, `renderStrip()` and its call in `boot()`.
  - Add `import { initPipeline, refreshRuns } from "./pipeline.js";`.
  - In `createRun()`, after `setStatus(...)`, add `refreshRuns(run.id);`.
  - In `boot()`, after `await initAgent();`, add `await initPipeline();`.

- [ ] **Step 4: Styles.** Append to `frontend/css/station.css`:

```css
.runbar { align-items: center; gap: 8px; padding: 8px 12px; }
.runbar select { min-width: 420px; }
.stage.ready { border-top-color: var(--blue); }
.form .help { font-size: 11px; color: var(--muted); margin: -4px 0 6px; }
.files { display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }
.files img, .files video { max-width: 520px; border: 1px solid var(--line); }
```

- [ ] **Step 5: Verify.**
  - `/home/ubuntu/miniforge3/envs/node/bin/node --input-type=module --check < frontend/js/pipeline.js`, and the same for `app.js`: no output.
  - `uv run pytest -q`: all pass.
  - In a browser, run `make dev` and `make worker-robosuite`:
    1. In PROJECT, click CREATE RUN; the PIPELINE run selector shows the new run.
    2. The strip shows six stages. CONVERT is grey with the tooltip "no online worker with capability 'convert'".
    3. Click SCENE PREVIEW, set n = 2, RUN STAGE. The log streams, the state becomes DONE, and `grid.png` shows under OUTPUTS.
    4. Click ROLLOUT, keep policy `expert`, RUN STAGE. A rollout video and `rollout.json` appear.
  - Stop the servers (kill by PID).

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat(frontend): PIPELINE tab with run selector, stage readiness, params, live log and outputs"
```

---

### Task 7: Stage gates, Agent Brief (function, API, CLI) and five extension templates

**Files:**
- Create:
  - `harness/gates.py`, `harness/brief.py`;
  - `harness/gates/{engine,scene,objects,expert,data_method,convert,intel,train,eval,rollout,assist,report}.yaml`;
  - `harness/tasks/{add_engine,add_object_importer,add_data_method,add_eval_metric,add_report_section}.yaml`;
  - `tests/harness/test_gates_brief.py`.
- Modify:
  - `harness/model.py` (`TaskTemplate.gate: str = ""`);
  - the 7 existing `harness/tasks/*.yaml` (add `gate:`);
  - `harness/api.py` (`GET /gates`, `POST /brief`);
  - `harness/__main__.py` (`brief` subcommand);
  - `configs/catalog.yaml` (the `mujoco` item points to `add_engine`; new `newton` item).

**Interfaces:**
- Consumes: `engines.registry.load_engines`, `EngineManifest` (Task 1); `harness.model.TaskTemplate`; `Harness.templates: dict[str, TaskTemplate]`.
- Produces:
  - `harness.gates`:
    - `GATES_DIR`, `GATE_IDS` (the 12 ids in pipeline order);
    - `Gate(id, title, status: ready|partial|planned, owner, extension_point, contract: list[str], conformance: str, paths: list[str], reference=[], context=[], templates=[], conventions=[])`;
    - `load_gates(gates_dir=GATES_DIR) -> dict[str, Gate]`.
  - `harness.brief.build_brief(request: str, stage_ids: list[str], engine_ids: list[str], gates, templates, engines) -> str`. It is deterministic, raises `ValueError` on an empty request or unknown stage, and accepts unknown engines (described as "new engine").
  - `harness.brief.brief_paths(stage_ids, gates) -> list[str]`: the union of the gates' `paths`, in order. Used as `free_task.paths` by Task 8.
  - HTTP:
    - `GET /api/agent/gates` → list of gate dicts;
    - `POST /api/agent/brief {request, stages, engines}` → `{markdown, paths}`, or 400 on `ValueError`.
  - CLI: `python -m harness brief "<request>" [--stage S ...] [--engine E ...] [--out file.md]`.

- [ ] **Step 1: Write the failing tests** in `tests/harness/test_gates_brief.py`:

```python
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from engines.registry import load_engines
from harness.__main__ import main
from harness.brief import brief_paths, build_brief
from harness.gates import GATE_IDS, load_gates
from harness.model import load_templates
from harness.server import create_agent_app
from tests.harness.conftest import make_harness

ROOT = Path(__file__).resolve().parents[2]
GATES = load_gates()
TEMPLATES = load_templates(ROOT / "harness/tasks")


def test_every_stage_has_a_valid_gate():
    assert set(GATES) == set(GATE_IDS)
    for g in GATES.values():
        assert g.contract and g.conformance and g.paths, g.id
        for rel in g.reference:
            assert (ROOT / rel).exists(), f"{g.id}: missing reference {rel}"


def test_templates_and_gates_point_at_each_other():
    for t in TEMPLATES.values():
        if t.id != "free_task":
            assert t.gate in GATES, f"template {t.id} has no valid gate"
    for g in GATES.values():
        for tid in g.templates:
            assert tid in TEMPLATES, f"gate {g.id} lists unknown template {tid}"


def test_brief_is_deterministic_and_complete():
    args = ("Add a Newton engine with the Panda and our table scene", ["eval", "engine"], ["newton", "robosuite"],
            GATES, TEMPLATES, load_engines())
    md = build_brief(*args)
    assert md == build_brief(*args)
    assert md.startswith("# Agent brief: Add a Newton engine")
    assert md.index("(`engine`") < md.index("(`eval`")                  # pipeline order, not request order
    assert GATES["engine"].contract[0] in md and GATES["engine"].conformance in md
    assert "python -m harness run add_engine" in md
    assert "`newton`: new engine" in md and "`robosuite` (" in md
    assert "Co-authored-by" in md and "cathlab_v1.yaml" in md


def test_brief_errors_and_paths():
    with pytest.raises(ValueError, match="unknown stage"):
        build_brief("x", ["warp"], [], GATES, TEMPLATES, {})
    with pytest.raises(ValueError, match="empty"):
        build_brief("  ", [], [], GATES, TEMPLATES, {})
    paths = brief_paths(["engine", "eval"], GATES)
    assert paths[0] == GATES["engine"].paths[0] and len(paths) == len(set(paths))


def test_brief_api_and_gates(tmp_path, repo):
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    assert len(c.get("/api/agent/gates").json()) == len(GATE_IDS)
    r = c.post("/api/agent/brief", json={"request": "Add ManiSkill", "stages": ["engine"], "engines": ["maniskill"]})
    assert r.status_code == 200 and "# Agent brief: Add ManiSkill" in r.json()["markdown"]
    assert r.json()["paths"] == GATES["engine"].paths
    assert c.post("/api/agent/brief", json={"request": "x", "stages": ["nope"]}).status_code == 400


def test_brief_cli_writes_file(tmp_path):
    out = tmp_path / "brief.md"           # `brief` reads templates and gates only; it never touches git
    rc = main(["--repo", str(ROOT), "--no-fetch", "brief", "Add a metric", "--stage", "eval", "--out", str(out)])
    assert rc == 0 and out.read_text().startswith("# Agent brief: Add a metric")
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/harness/test_gates_brief.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'harness.brief'`.

- [ ] **Step 3: Implement `harness/gates.py`:**

```python
"""Stage gates: for each pipeline stage, where an extension goes, the contract it meets and how it is checked."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

GATES_DIR = Path(__file__).resolve().parent / "gates"
GATE_IDS = ("engine", "scene", "objects", "expert", "data_method", "convert", "intel", "train", "eval", "rollout",
            "assist", "report")


class Gate(BaseModel):
    id: str
    title: str
    status: Literal["ready", "partial", "planned"]
    owner: str
    extension_point: str
    contract: list[str]
    conformance: str
    paths: list[str]                      # suggested allowed paths for an agent job on this stage
    reference: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)


def load_gates(gates_dir: str | Path = GATES_DIR) -> dict[str, Gate]:
    gates: dict[str, Gate] = {}
    for p in sorted(Path(gates_dir).glob("*.yaml")):
        gate = Gate.model_validate(yaml.safe_load(p.read_text()))
        if gate.id != p.stem:
            raise ValueError(f"{p}: id '{gate.id}' must equal the file name")
        gates[gate.id] = gate
    return gates
```

- [ ] **Step 4: Write the 12 gate files.** They are content, so reproduce them exactly. Every `reference` must exist on `main`.

`harness/gates/engine.yaml`:

```yaml
id: engine
title: Simulation engine plugin
status: partial
owner: "Rakesh (robosuite, MuJoCo, Newton), Navneet (Isaac)"
extension_point: "engines/<id>/ with engine.yaml + cli.py (+ a SimBackend and an ExpertSim adapter). One folder per simulator."
contract:
  - "engines/<id>/engine.yaml validates as engines.registry.EngineManifest: id = folder name, io = a profile in engines.registry.IO_PROFILES, capabilities per stage ready|in_progress|planned"
  - "python -m engines.<id>.cli {preview|check-expert|generate|convert|eval|rollout} --out DIR [--run run.yaml], built with engines.cli_base.run_cli"
  - "Exit 0 on success, non-zero with a one-line reason on stderr; outputs only under --out; simulator imports are lazy (--help works in the core env)"
  - "manifest.backend = 'module:Class' implementing contracts.SimBackend with the io profile's obs keys and shapes; StepInfo.stage_flags over contracts.STAGES"
  - "An ExpertSim adapter (dataengine.expert.ExpertSim) so the shared waypoint expert and engines.expert_policy.ExpertPolicy run unchanged"
conformance: "python -m pytest -q tests/engines/test_conformance.py -k <id>   # in the engine's env"
paths: ["engines/", "sim/", "tests/engines/", "tests/sim/", "configs/catalog.yaml", "docs/decisions.md", "CREDITS.md"]
reference: [engines/robosuite/engine.yaml, engines/robosuite/cli.py, engines/robosuite/evaluate.py, sim/robosuite/expert_sim.py, sim/robosuite/backend.py]
context: [AGENTS.md, docs/EXTENDING.md, engines/registry.py, engines/cli_base.py, contracts/sim.py, dataengine/expert.py, docs/specs/2026-10-23-multi-engine-platform-design.md]
templates: [add_engine, add_sim_backend]
conventions:
  - "Core frame: robot base, table top z = 0, quaternions wxyz (sim/robosuite/README.md)"
  - "The same EpisodeConfig seed is the same episode in every engine (dataengine.variation)"
  - "Flip a capability to ready only after its check passes; record the evidence in docs/decisions.md"
```

`harness/gates/scene.yaml`:

```yaml
id: scene
title: Scene and variation axes
status: ready
owner: Rakesh
extension_point: "A new versioned spec configs/variation/cathlab_v<N>.yaml + dataengine/variation.py axes + the engine's randomizer (sim/<engine>/randomize.py)"
contract:
  - "dataengine.variation.VariationSpec loads the spec; train_configs/eval_configs return EpisodeConfigs deterministically from seeds"
  - "Eval tiers stay dataengine.variation.TIERS; a new axis gets held-out values for the all_held_out tier"
  - "Never edit configs/variation/cathlab_v1.yaml (the published v1 dataset depends on it)"
conformance: "uv run pytest -q tests/dataengine && MUJOCO_GL=egl conda run -n robosuite python scripts/robosuite_expert_check.py --n 20"
paths: ["configs/variation/", "dataengine/variation.py", "sim/", "tests/dataengine/", "tests/sim/", "configs/catalog.yaml", "docs/decisions.md"]
reference: [configs/variation/cathlab_v1.yaml, dataengine/variation.py, sim/robosuite/randomize.py]
context: [AGENTS.md, docs/EXTENDING.md, docs/specs/2026-10-22-sim-framework-design.md]
templates: [add_variation_axis]
```

`harness/gates/objects.yaml`:

```yaml
id: objects
title: Objects on the table
status: partial
owner: Rakesh
extension_point: "Primitive objects in a new VariationSpec version (objects: shape, dims, mass, grasp); mesh/Objaverse object packages assets/objects/<name>/ + assets/importers/ (Plan B-engines)"
contract:
  - "Every object has a grasp template the analytic expert understands (dataengine.expert.grasp_pose)"
  - "Object packages: object.yaml + visual.obj + collision/*.obj + generated object.xml (MJCF) that every engine can load"
  - "Importers validate meshes (watertight warning, scale bounds, hull count limit) and never overwrite an existing package"
conformance: "MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli check-expert --out var/check --n 20 --objects <name>   # rate >= 0.8"
paths: ["configs/variation/", "assets/", "dataengine/", "sim/", "tests/", "configs/catalog.yaml", "docs/decisions.md", "CREDITS.md"]
reference: [configs/variation/cathlab_v1.yaml, dataengine/expert.py, sim/robosuite/env.py]
context: [AGENTS.md, docs/EXTENDING.md, docs/specs/2026-10-23-multi-engine-platform-design.md]
templates: [add_object, add_object_importer]
```

`harness/gates/expert.yaml`:

```yaml
id: expert
title: Expert controller
status: ready
owner: Rakesh
extension_point: "dataengine/expert.py (shared waypoint expert, ExpertParams) + an optional grasp provider in dataengine/experts/<id>.py; per-engine ExpertSim adapters"
contract:
  - "dataengine.expert.run_expert_episode(sim, spec, cfg, params) works with any ExpertSim"
  - "A grasp provider returns (tcp_position, tcp_quat_wxyz, approach) like dataengine.expert.grasp_pose; heavy models are imported lazily"
  - "check-expert writes expert_check.json: {split, params, per_object: {obj: {success, attempted, rate, mean_steps, max_steps}}, failures}"
conformance: "MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli check-expert --out var/check --n 20   # every rate >= 0.8"
paths: ["dataengine/", "scripts/robosuite_expert_check.py", "scripts/robosuite_generate.py", "engines/", "tests/", "configs/catalog.yaml", "docs/decisions.md", "CREDITS.md"]
reference: [dataengine/expert.py, sim/robosuite/expert_sim.py, scripts/robosuite_expert_check.py, engines/expert_policy.py]
context: [AGENTS.md, docs/EXTENDING.md, dataengine/variation.py]
templates: [add_expert]
```

`harness/gates/data_method.yaml`:

```yaml
id: data_method
title: Synthetic data method
status: partial
owner: "Rakesh (scripted), team (MimicGen, Cosmos, teleop)"
extension_point: "dataengine/registry.py (@register_method) + dataengine/<method>.py; the engine's generate stage for engine-side methods"
contract:
  - "Raw output per shard: <out>/raw/shard_<i>-of-<k>/episodes/*.npz + meta/episodes_extra.jsonl + manifest.yaml (contracts.DatasetManifest with the right source)"
  - "Episodes keep the io profile of their engine (panda_libero: action 7, state 8, agentview + eye_in_hand 256x256)"
  - "Only successful episodes are kept; the episode's EpisodeConfig is stored in episodes_extra"
conformance: "uv run pytest -q tests/dataengine"
paths: ["dataengine/", "scripts/", "engines/", "tests/dataengine/", "configs/catalog.yaml", "docs/decisions.md", "docs/data.md", "CREDITS.md"]
reference: [dataengine/registry.py, dataengine/generators.py, dataengine/mimicgen.py, dataengine/npz_sink.py, scripts/robosuite_generate.py]
context: [AGENTS.md, docs/EXTENDING.md, docs/data.md, contracts/specs.py, docs/research]
templates: [extend_dataset, add_data_method]
```

`harness/gates/convert.yaml`:

```yaml
id: convert
title: LeRobot conversion
status: ready
owner: Rakesh
extension_point: "scripts/npz_to_lerobot.py (panda_libero) or the engine CLI's convert stage (runs in the lerobot env)"
contract:
  - "LeRobot v3 dataset in <out>/lerobot with the io profile's features and fps; episodes re-indexed from 0"
  - "meta/episodes_extra.jsonl and manifest.yaml copied; tags v<N> + v3.0 when pushed to the Hub"
conformance: "uv run pytest -q tests/dataengine/test_npz_sink.py tests/dataengine/test_lerobot_sink.py"
paths: ["scripts/npz_to_lerobot.py", "dataengine/", "engines/", "tests/dataengine/", "docs/data.md"]
reference: [scripts/npz_to_lerobot.py, dataengine/npz_sink.py, dataengine/lerobot_sink.py]
context: [AGENTS.md, docs/data.md]
templates: [extend_dataset]
```

`harness/gates/intel.yaml`:

```yaml
id: intel
title: Dataset intelligence
status: planned
owner: Sharan
extension_point: "dataset_intel/ (validators, coverage over the variation grid, readiness score)"
contract:
  - "IntelReport as defined in docs/handoffs/2026-10-23-sharan-serve-intel-session.md (S5)"
  - "Runs on a LeRobot dataset folder without a GPU; writes intel.json + intel.md"
conformance: "uv run pytest -q tests/dataset_intel"
paths: ["dataset_intel/", "tests/dataset_intel/", "docs/decisions.md"]
reference: [docs/handoffs/2026-10-23-sharan-serve-intel-session.md, dataengine/variation.py]
context: [AGENTS.md, docs/EXTENDING.md, docs/data.md]
templates: []
```

`harness/gates/train.yaml`:

```yaml
id: train
title: Policy training
status: partial
owner: Rakesh
extension_point: "train/recipes/<model>.yaml + a PolicyAdapter for the model (lerobot env, GPU worker)"
contract:
  - "contracts.PolicyAdapter: load(ckpt), predict(obs, instruction) -> (chunk, action_dim) float32, meta()"
  - "Trained checkpoints go to the Hub as <owner>/<model>_<embodiment>_pick_v<N> with a contracts.ModelCard"
conformance: "uv run pytest -q tests/test_contracts.py"
paths: ["train/", "tests/", "configs/catalog.yaml", "docs/decisions.md", "CREDITS.md"]
reference: [contracts/policy.py, docs/handoffs/2026-10-23-training-gpu-session.md]
context: [AGENTS.md, docs/EXTENDING.md, contracts/reports.py]
templates: [add_policy_adapter]
```

`harness/gates/eval.yaml`:

```yaml
id: eval
title: Evaluation
status: partial
owner: "Rakesh (engine eval stages), Akanksh (metrics, VLM judge)"
extension_point: "eval/ (metrics, judge, failure analysis) and each engine's eval stage; suites in configs/eval/"
contract:
  - "contracts.EvalReport with one EpisodeResult per episode; scene_variant = variation tier; failed episodes keep a video_path"
  - "Suites: objects x tiers x seeds (configs/eval/suite_v2.yaml); seeds >= 5000; the expert is the reference row"
  - "Metrics read EvalReport JSON only, so they work for every engine"
conformance: "uv run pytest -q tests/test_metrics.py tests/test_runner.py tests/test_failures.py tests/engines"
paths: ["eval/", "engines/", "configs/eval/", "tests/", "docs/decisions.md"]
reference: [engines/robosuite/evaluate.py, eval/metrics.py, eval/failures.py, contracts/reports.py, configs/eval/suite_v2.yaml]
context: [AGENTS.md, docs/EXTENDING.md, eval/README.md]
templates: [add_eval_metric]
```

`harness/gates/rollout.yaml`:

```yaml
id: rollout
title: Serving, registry and rollout
status: partial
owner: "Sharan (policy server, registry), Rakesh (engine rollout stages)"
extension_point: "serve/ (policy server + PolicyClient + registry with promote/rollback) and each engine's rollout stage"
contract:
  - "rollout writes rollout.json {success, steps, failed_stage, latency_ms_p50, latency_ms_p95, object, tier, seed, instruction, policy, video} + the video"
  - "--policy url:<policy server> uses serve.policy_client.PolicyClient (HTTP + JSON, X-Cathlab-Token)"
  - "Promotion rule and rollback as in docs/handoffs/2026-10-23-sharan-serve-intel-session.md (S3)"
conformance: "uv run pytest -q tests/engines tests/serve"
paths: ["serve/", "engines/", "tests/serve/", "tests/engines/", "docs/decisions.md"]
reference: [engines/robosuite/evaluate.py, engines/policies.py, docs/handoffs/2026-10-23-sharan-serve-intel-session.md]
context: [AGENTS.md, docs/EXTENDING.md, configs/serve.yaml]
templates: []
```

`harness/gates/assist.yaml`:

```yaml
id: assist
title: Awareness assistant (ASSIST)
status: partial
owner: "Navneet (awareness), Rakesh (panel)"
extension_point: "awareness/ (procedure model, VLM step tracking, Assistant) -> an instruction for a rollout (--instruction)"
contract:
  - "awareness.assistant.Assistant proposes the next object for the current procedure step; the user confirms; the instruction drives the production model's rollout"
  - "No network calls in tests; the VLM is behind an interface with a fake"
conformance: "uv run pytest -q tests/test_awareness_assistant.py tests/test_awareness_engine.py tests/test_awareness_procedure.py"
paths: ["awareness/", "configs/procedures/", "tests/", "docs/decisions.md"]
reference: [awareness/assistant.py, awareness/engine.py, awareness/procedure.py]
context: [AGENTS.md, awareness/README.md]
templates: []
```

`harness/gates/report.yaml`:

```yaml
id: report
title: Failure report and next run
status: planned
owner: Sharan
extension_point: "report/failure_report.py (+ sections) reading EvalReports"
contract:
  - "report.md + report.json + next_run.json as defined in docs/handoffs/2026-10-23-sharan-serve-intel-session.md (S6)"
  - "next_run.json lists exactly the failing cells (engine x tier x object x layout/colour) as a run selection the Control Station can create"
conformance: "uv run pytest -q tests/report"
paths: ["report/", "tests/report/", "docs/decisions.md"]
reference: [eval/failures.py, docs/handoffs/2026-10-23-sharan-serve-intel-session.md]
context: [AGENTS.md, docs/EXTENDING.md, contracts/reports.py, contracts/gaps.py]
templates: [add_report_section]
```

- [ ] **Step 5: Implement `harness/brief.py`:**

```python
"""Agent Brief: a deterministic markdown brief for any coding agent, from a request + the stage gates it touches."""
from engines.registry import EngineManifest
from harness.gates import GATE_IDS, Gate
from harness.model import TaskTemplate

RULES = (
    "Read AGENTS.md first and follow it.",
    "Branch from the tip of origin/main; small Conventional Commits.",
    "Commits are authored by the human only: no Co-authored-by trailers for AI agents.",
    "Never commit data, checkpoints, .env or secrets; outputs go under var/ or data/ (git-ignored).",
    "Never edit configs/variation/cathlab_v1.yaml (the published v1 dataset depends on it); add a new versioned spec.",
    "Add a row to CREDITS.md for every project, model, dataset or paper you use.",
    "Record decisions and evidence (check results) in docs/decisions.md.",
)


def _ordered(stage_ids: list[str], gates: dict[str, Gate]) -> list[Gate]:
    unknown = sorted(set(stage_ids) - set(gates))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected some of {list(GATE_IDS)}")
    order = {g: i for i, g in enumerate(GATE_IDS)}
    return [gates[s] for s in sorted(set(stage_ids), key=lambda s: (order.get(s, len(order)), s))]


def brief_paths(stage_ids: list[str], gates: dict[str, Gate]) -> list[str]:
    paths: list[str] = []
    for gate in _ordered(stage_ids, gates):
        paths += [p for p in gate.paths if p not in paths]
    return paths


def _capabilities(m: EngineManifest) -> list[tuple[str, str]]:
    flat = []
    for key, value in m.capabilities.items():
        flat += [(key, value)] if isinstance(value, str) else [(f"{key}.{k}", v) for k, v in value.items()]
    return flat


def build_brief(request: str, stage_ids: list[str], engine_ids: list[str], gates: dict[str, Gate],
                templates: dict[str, TaskTemplate], engines: dict[str, EngineManifest]) -> str:
    if not request.strip():
        raise ValueError("request is empty")
    selected = _ordered(stage_ids, gates)
    title = request.strip().splitlines()[0][:80]
    lines = [f"# Agent brief: {title}", "", "## Request", "", request.strip(), "", "## Repository rules", ""]
    lines += [f"- {rule}" for rule in RULES]
    for g in selected:
        lines += ["", f"## Stage gate: {g.title} (`{g.id}`, {g.status}, owner {g.owner})", "",
                  f"**Extension point:** {g.extension_point}", "", "**Contract:**", ""]
        lines += [f"- {c}" for c in g.contract]
        lines += ["", f"**Conformance (must pass):** `{g.conformance}`", "",
                  "**Allowed paths (suggested):** " + ", ".join(f"`{p}`" for p in g.paths)]
        if g.reference:
            lines += ["", "**Reference implementation:** " + ", ".join(f"`{r}`" for r in g.reference)]
        if g.context:
            lines += ["", "**Read first:** " + ", ".join(f"`{c}`" for c in g.context)]
        if g.conventions:
            lines += ["", "**Conventions:**", ""] + [f"- {c}" for c in g.conventions]
        matching = [templates[t] for t in g.templates if t in templates]
        if matching:
            lines += ["", "**Matching harness templates** (UI: AGENT tab):", ""]
            for t in matching:
                params = " ".join(f"--param {p.name}=..." for p in t.params if p.required)
                cmd = f"python -m harness run {t.id} {params}".rstrip()
                lines.append(f"- `{cmd}`: {t.title}")
    if engine_ids:
        lines += ["", "## Engines", ""]
        for eid in sorted(set(engine_ids)):
            m = engines.get(eid)
            if m is None:
                lines.append(f"- `{eid}`: new engine, not in engines/ yet. Follow the `engine` gate; copy engines/robosuite/.")
                continue
            pending = [f"{k} ({v})" for k, v in _capabilities(m) if v != "ready"]
            env = m.env.name or m.env.python or ""
            lines.append(f"- `{eid}` ({m.label}): env {m.env.kind} {env}, worker `{m.worker}`, io `{m.io}`; "
                         f"not ready: {', '.join(pending) or 'none'}")
    lines += ["", "## Commands", "", "```bash", "uv run pytest -q          # core tests stay green",
              "make dev                  # Control Station on http://localhost:8700"]
    lines += sorted({g.conformance for g in selected})
    lines += ["```", "", "## Done when", "",
              "- The conformance commands above pass and the new behaviour has tests.",
              "- Status flags change only for what now works (configs/catalog.yaml, engines/<id>/engine.yaml).",
              "- docs/decisions.md has a row with the evidence; CREDITS.md lists what you used.", ""]
    return "\n".join(lines)
```

- [ ] **Step 6: Wire up the template model, API and CLI.**
  - **`harness/model.py`:** add `gate: str = ""` to `TaskTemplate` after `catalog_target`.
  - **Existing templates:** add a `gate:` line after `catalog_target:`:
    - `add_object: objects`, `extend_dataset: data_method`, `add_variation_axis: scene`, `add_sim_backend: engine`, `add_policy_adapter: train`, `add_expert: expert`;
    - `free_task: ""` (leave it out).
  - **`harness/api.py`:**

```python
from engines.registry import load_engines
from harness.brief import brief_paths, build_brief
from harness.gates import load_gates


class BriefRequest(BaseModel):
    request: str
    stages: list[str] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=list)
```

Inside `create_agent_router`:

```python
    @r.get("/gates")
    def gates() -> list[dict]:
        return [g.model_dump() for g in load_gates().values()]

    @r.post("/brief")
    def brief(body: BriefRequest) -> dict:
        gates = load_gates()
        try:
            md = build_brief(body.request, body.stages, body.engines, gates, harness.templates, load_engines())
            return {"markdown": md, "paths": brief_paths(body.stages, gates)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
```

  - **`harness/__main__.py`:** add the subparser:

```python
    brief = sub.add_parser("brief", help="print an Agent Brief for a request")
    brief.add_argument("request")
    brief.add_argument("--stage", action="append", default=[], help="gate id (repeatable)")
    brief.add_argument("--engine", action="append", default=[], help="engine id (repeatable)")
    brief.add_argument("--out", help="write the markdown here instead of stdout")
```

and the branch:

```python
    elif args.cmd == "brief":
        from engines.registry import load_engines
        from harness.brief import build_brief
        from harness.gates import load_gates
        md = build_brief(args.request, args.stage, args.engine, load_gates(), h.templates, load_engines())
        if args.out:
            Path(args.out).write_text(md)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(md)
```

  (Import `Path` if `__main__.py` doesn't already.)

- [ ] **Step 7: Write the five new templates.** They follow the existing template style; `{param}` placeholders are rendered by `harness.model.render_prompt`.

`harness/tasks/add_engine.yaml`:

```yaml
id: add_engine
title: Add a simulation engine plugin
description: >
  Add engines/<engine_id>/ (manifest + uniform CLI + SimBackend + ExpertSim adapter) so the Control Station can
  preview, generate, evaluate and roll out in a new simulator, following the robosuite reference.
catalog_target: simulator
gate: engine
params:
  - {name: engine_id, type: str, pattern: "^[a-z][a-z0-9_]*$", help: "folder name, e.g. mujoco, newton, maniskill"}
  - {name: label, type: str, help: "human label, e.g. 'MuJoCo 3.3 standalone (Menagerie Panda)'"}
  - {name: env, type: str, default: "", required: false, help: "conda env / venv to run it in"}
  - {name: notes, type: text, required: false, help: "simulator version, robot asset source, known constraints"}
context:
  - AGENTS.md
  - docs/EXTENDING.md
  - harness/gates/engine.yaml
  - engines/registry.py
  - engines/cli_base.py
  - engines/robosuite/engine.yaml
  - engines/robosuite/cli.py
  - engines/robosuite/evaluate.py
  - sim/robosuite/expert_sim.py
  - sim/robosuite/backend.py
prompt: |
  Add the engine plugin `{engine_id}` ({label}), env `{env}`. Notes: {notes}
  Follow harness/gates/engine.yaml exactly and mirror engines/robosuite/:
  1. engines/{engine_id}/engine.yaml (capabilities honest: only what works is `ready`), cli.py with the six stages
     via engines.cli_base.run_cli, lazy simulator imports.
  2. sim/{engine_id}/: ExpertSim adapter (core frame) and SimBackend (io profile panda_libero unless told otherwise).
  3. tests/engines/: CLI dry-run tests runnable in the core env; the conformance test must pass in the engine's env.
  4. Catalog item `{engine_id}` in section simulator; CREDITS.md rows; one docs/decisions.md row with the
     conformance evidence.
allowed_paths:
  - "engines/{engine_id}/"
  - "sim/{engine_id}/"
  - tests/engines/
  - configs/catalog.yaml
  - docs/decisions.md
  - CREDITS.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 90
```

`harness/tasks/add_object_importer.yaml`:

```yaml
id: add_object_importer
title: Add an object importer
description: >
  Add an importer that turns a source (primitive dims, mesh file, Objaverse search) into an object package
  assets/objects/<name>/ every engine can load.
catalog_target: objects
gate: objects
params:
  - {name: importer_id, type: str, pattern: "^[a-z][a-z0-9_]*$", help: "e.g. mesh, objaverse"}
  - {name: notes, type: text, required: false, help: "source format, libraries (trimesh, CoACD), limits"}
context:
  - AGENTS.md
  - docs/EXTENDING.md
  - harness/gates/objects.yaml
  - configs/variation/cathlab_v1.yaml
  - dataengine/expert.py
prompt: |
  Add the object importer `{importer_id}`. Notes: {notes}
  Follow harness/gates/objects.yaml: assets/importers/{importer_id}.py producing assets/objects/<name>/ (object.yaml,
  visual.obj, collision/*.obj, object.xml), validation (watertight warning, scale bounds, hull limit), refuse to
  overwrite. Heavy libraries imported lazily. Tests in tests/assets/ with a tiny generated cube. CREDITS.md rows.
allowed_paths:
  - assets/
  - tests/assets/
  - pyproject.toml
  - docs/decisions.md
  - CREDITS.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 60
```

`harness/tasks/add_data_method.yaml`:

```yaml
id: add_data_method
title: Add a synthetic data method
description: >
  Add a data generation method (e.g. MimicGen augmentation, Cosmos transfer, teleop import) behind the generator
  registry, writing raw shards in the shared format.
catalog_target: ""
gate: data_method
params:
  - {name: method_id, type: str, pattern: "^[a-z][a-z0-9_]*$", help: "e.g. mimicgen, cosmos_transfer"}
  - {name: notes, type: text, required: false, help: "paper/repo, GPU needs, inputs"}
context:
  - AGENTS.md
  - docs/EXTENDING.md
  - harness/gates/data_method.yaml
  - dataengine/registry.py
  - dataengine/generators.py
  - dataengine/npz_sink.py
  - docs/data.md
prompt: |
  Add the data method `{method_id}`. Notes: {notes}
  Follow harness/gates/data_method.yaml: dataengine/{method_id}.py registered with @register_method("{method_id}"),
  raw shards + DatasetManifest (source set correctly), only successful episodes, lazy heavy imports, tests in
  tests/dataengine/ with the fake sim. CREDITS.md row for the method's paper/repo; docs/data.md section.
allowed_paths:
  - dataengine/
  - tests/dataengine/
  - docs/data.md
  - docs/decisions.md
  - CREDITS.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 60
```

`harness/tasks/add_eval_metric.yaml`:

```yaml
id: add_eval_metric
title: Add an evaluation metric
description: >
  Add a metric computed from EvalReport JSON (any engine), shown in reports and usable by the promotion rule.
catalog_target: ""
gate: eval
params:
  - {name: metric_id, type: str, pattern: "^[a-z][a-z0-9_]*$", help: "e.g. wrong_object_rate, time_to_grasp"}
  - {name: definition, type: text, help: "exact definition, edge cases"}
context:
  - AGENTS.md
  - docs/EXTENDING.md
  - harness/gates/eval.yaml
  - contracts/reports.py
  - eval/metrics.py
prompt: |
  Add the metric `{metric_id}`: {definition}
  Follow harness/gates/eval.yaml: a pure function in eval/metrics.py over contracts.EvalReport, tests in
  tests/test_metrics.py with hand-built reports (including empty and all-failed), and a line in eval/README.md.
allowed_paths:
  - eval/
  - tests/test_metrics.py
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 30
```

`harness/tasks/add_report_section.yaml`:

```yaml
id: add_report_section
title: Add a failure report section
description: >
  Add a section to the failure report (e.g. sim2sim gap, wrong-object grasps) with its recommendation for the next run.
catalog_target: ""
gate: report
params:
  - {name: section_id, type: str, pattern: "^[a-z][a-z0-9_]*$", help: "e.g. sim2sim_gap"}
  - {name: definition, type: text, help: "what it shows and which next-run cells it recommends"}
context:
  - AGENTS.md
  - docs/EXTENDING.md
  - harness/gates/report.yaml
  - contracts/reports.py
  - eval/failures.py
prompt: |
  Add the failure report section `{section_id}`: {definition}
  Follow harness/gates/report.yaml: a function in report/ reading EvalReports, rendered into report.md/report.json,
  contributing failing cells to next_run.json; tests in tests/report/ with hand-built reports.
allowed_paths:
  - report/
  - tests/report/
  - docs/decisions.md
acceptance:
  - {name: core-tests, cmd: "uv run pytest -q"}
timeout_min: 45
```

**Catalog** (`configs/catalog.yaml`, section `simulator`):
- Set the `mujoco` item's `harness_task` to `add_engine`.
- Add: `- {id: newton, label: "Newton 1.6 (Warp, GPU)", status: planned, harness_task: add_engine}`.

**Frontend prefill** (`frontend/js/app.js`, `buildWithAgent`): add `if (taskId === "add_engine" && itemId) prefill.engine_id = itemId;`.

- [ ] **Step 8: Run all tests**

Run: `uv run pytest -q`. All pass, including `tests/harness/test_templates.py`: 12 templates load and render, and every catalog `harness_task` exists.

- [ ] **Step 9: Commit**

```bash
git add harness configs/catalog.yaml frontend/js/app.js tests/harness/test_gates_brief.py
git commit -m "feat(harness): stage gates, deterministic Agent Brief (API + CLI) and five extension templates"
```

---

### Task 8: Agent Brief drawer on every tab

**Files:**
- Create: `frontend/js/brief.js`
- Modify: `frontend/index.html` (BRIEF header button + drawer), `frontend/js/app.js` (init), `frontend/js/pipeline.js` (an AGENT BRIEF button in the stage detail), `frontend/css/station.css`

**Interfaces:**
- Consumes:
  - `GET /api/agent/gates`, `POST /api/agent/brief` → `{markdown, paths}` (Task 7); `GET /api/engines` (Task 5);
  - `openTemplate(taskId, prefill)` from `agent.js`; `showTab` from `app.js`; `stageGate` from `pipeline.js`.
- Produces: `export async function initBrief()`, `export function openBrief({ stages = [], engines = [], request = "" } = {})`.

- [ ] **Step 1: Markup.** In `frontend/index.html`:
  - In the header after `<nav …>`, add `<button class="btn brief" id="btn-brief" title="Agent Brief: describe anything the UI doesn't offer">AGENT BRIEF</button>`.
  - Before `<footer>`, add:

```html
  <!-- AGENT BRIEF drawer (every tab) -->
  <aside class="drawer hidden" id="brief">
    <div class="panel-title">AGENT BRIEF</div>
    <div class="muted">Describe what you need. The brief collects the stage contracts, tests, templates and rules for
      any coding agent. Copy it, download it, or run it with the harness.</div>
    <textarea id="brief-request" class="field" rows="5"
      placeholder="e.g. Add a ManiSkill engine with the Panda and our three objects"></textarea>
    <div class="panel-title">STAGES</div>
    <div id="brief-stages" class="checks"></div>
    <div class="panel-title">ENGINES</div>
    <div id="brief-engines" class="checks"></div>
    <input id="brief-new-engine" class="field" placeholder="new engine id (optional), e.g. maniskill">
    <div class="row">
      <button class="btn primary" id="brief-build">BUILD BRIEF</button>
      <button class="btn" id="brief-copy" disabled>COPY</button>
      <button class="btn" id="brief-download" disabled>DOWNLOAD .MD</button>
      <button class="btn agent" id="brief-run" disabled>RUN WITH AGENT</button>
      <button class="btn" id="brief-close">CLOSE</button>
    </div>
    <pre class="console" id="brief-out"></pre>
  </aside>
```

- [ ] **Step 2: Create `frontend/js/brief.js`:**

```javascript
// AGENT BRIEF drawer: request + stage/engine ticks -> deterministic markdown brief -> copy / download / run.
import { api } from "./api.js";
import { $, el, setStatus } from "./dom.js";
import { openTemplate } from "./agent.js";

const bs = { markdown: "", paths: [] };

function checkbox(value, label, title) {
  const wrap = el("label", "check");
  const box = el("input");
  box.type = "checkbox";
  box.value = value;
  wrap.title = title || "";
  wrap.append(box, document.createTextNode(` ${label}`));
  return wrap;
}

function ticked(sel) { return [...document.querySelectorAll(`${sel} input:checked`)].map((b) => b.value); }

export async function initBrief() {
  const [gates, engines] = await Promise.all([api.get("/api/agent/gates").catch(() => []),
    api.get("/api/engines").catch(() => [])]);
  const stages = $("#brief-stages");
  for (const g of gates) stages.append(checkbox(g.id, g.title, `${g.status} · ${g.owner}`));
  const eng = $("#brief-engines");
  for (const e of engines) eng.append(checkbox(e.id, e.id, e.label));
  $("#btn-brief").addEventListener("click", () => openBrief());
  $("#brief-close").addEventListener("click", () => $("#brief").classList.add("hidden"));
  $("#brief-build").addEventListener("click", build);
  $("#brief-copy").addEventListener("click", async () => {
    await navigator.clipboard.writeText(bs.markdown);
    setStatus("BRIEF COPIED");
  });
  $("#brief-download").addEventListener("click", () => {
    const a = el("a");
    a.href = URL.createObjectURL(new Blob([bs.markdown], { type: "text/markdown" }));
    a.download = "agent-brief.md";
    a.click();
    URL.revokeObjectURL(a.href);
  });
  $("#brief-run").addEventListener("click", () => {
    $("#brief").classList.add("hidden");
    document.querySelector('#tabs button[data-tab="agent"]').click();
    openTemplate("free_task", { goal: bs.markdown, paths: bs.paths.join(", "), check: "uv run pytest -q" });
  });
}

export function openBrief({ stages = [], engines = [], request = "" } = {}) {
  document.querySelectorAll("#brief-stages input").forEach((b) => { b.checked = stages.includes(b.value); });
  document.querySelectorAll("#brief-engines input").forEach((b) => { b.checked = engines.includes(b.value); });
  if (request) $("#brief-request").value = request;
  $("#brief").classList.remove("hidden");
  $("#brief-request").focus();
}

async function build() {
  const engines = ticked("#brief-engines");
  const extra = $("#brief-new-engine").value.trim();
  if (extra) engines.push(extra);
  try {
    const r = await api.post("/api/agent/brief", { request: $("#brief-request").value, stages: ticked("#brief-stages"),
      engines });
    bs.markdown = r.markdown;
    bs.paths = r.paths;
    $("#brief-out").textContent = r.markdown;
    ["#brief-copy", "#brief-download", "#brief-run"].forEach((s) => { $(s).disabled = false; });
    setStatus("BRIEF READY");
  } catch (e) { setStatus(`BRIEF REJECTED: ${e.message}`); }
}
```

- [ ] **Step 3: Wire it up.**
  - **`app.js`:** `import { initBrief } from "./brief.js";` and in `boot()`, after `await initPipeline();`, add `await initBrief();`.
  - **`pipeline.js`:**
    - Add `import { openBrief } from "./brief.js";`.
    - Add `runs: [], engine: null` to the `ps` object.
    - In `refreshRuns`, after fetching, set `ps.runs = runs;`.
    - In `selectRun(id)`, set `ps.engine = ((ps.runs.find((r) => r.id === id) || {}).selection || {}).simulator || null;`.
    - In `renderDetail()`, right after the reason line (`box.append(el("div", s.reason ? …))`), add:

```javascript
  const gate = stageGate(s.id);
  if (gate) {
    const b = el("button", "btn agent", "AGENT BRIEF FOR THIS STAGE");
    b.type = "button";
    b.addEventListener("click", () => openBrief({ stages: [gate], engines: ps.engine ? [ps.engine] : [] }));
    box.append(b);
  }
```

- [ ] **Step 4: Styles.** Append to `station.css`:

```css
.drawer { position: fixed; top: 48px; right: 0; bottom: 44px; width: 560px; background: #fff;
  border-left: 2px solid var(--line); padding: 12px; overflow: auto; z-index: 10; display: flex;
  flex-direction: column; gap: 8px; box-shadow: -4px 0 12px rgba(0, 0, 0, 0.08); }
.drawer.hidden { display: none; }
.checks { display: flex; flex-wrap: wrap; gap: 4px 14px; }
.check { font-size: 13px; cursor: pointer; }
.btn.brief { margin-left: 12px; }
```

- [ ] **Step 5: Verify.**
  - Run `node --input-type=module --check` on `brief.js`, `pipeline.js` and `app.js` (node from the `node` conda env).
  - In a browser on `make dev`:
    1. From PROJECT, click AGENT BRIEF. Type "Add a ManiSkill engine", tick `Simulation engine plugin`, enter new engine `maniskill`, then BUILD BRIEF. The markdown shows the engine contract and `maniskill: new engine`.
    2. Click RUN WITH AGENT. The AGENT tab opens `free_task` with the goal = the brief and paths = `engines/, sim/, tests/engines/, …`.
    3. In PIPELINE, pick EVALUATE and click AGENT BRIEF FOR THIS STAGE. The drawer opens with the `eval` gate and the run's engine ticked.

- [ ] **Step 6: Commit**

```bash
git add frontend
git commit -m "feat(frontend): Agent Brief drawer on every tab with copy, download and run-with-agent"
```

---

### Task 9: Repo-wide agent instructions and docs

**Files:**
- Create:
  - `docs/EXTENDING.md`, `engines/README.md`;
  - `.github/instructions/engines.instructions.md`, `.github/instructions/objects.instructions.md`;
  - `.github/skills/cathlab-extend/SKILL.md`.
- Modify:
  - `AGENTS.md` ("Extending" section; ownership row `engines/`);
  - `docs/HARNESS.md` (12 templates, Agent Brief);
  - `docs/INSTALL.md` (pipeline needs workers; remote workers need the same repo path);
  - `README.md` (PIPELINE + Brief bullets); `Makefile` (`conformance-robosuite`);
  - `docs/handoffs/2026-10-23-isaac-engine-session.md` (use `engines.cli_base`; eval/rollout flags);
  - `docs/decisions.md` (one row).

- [ ] **Step 1: `docs/EXTENDING.md`** must contain these sections, each with concrete paths and commands taken from the gates. No placeholders; copy contract lines from `harness/gates/*.yaml`.
  1. **How the platform is put together:**
     - catalog → run (`var/runs/<id>/run.yaml`) → stages (`orchestrator/stages.py`) → engine CLI on a worker → outputs in `var/runs/<id>/<stage>/<job>/`;
     - a small ASCII diagram.
  2. **Stage gates:** a table of the 12 gates (id, title, status, owner, extension point, conformance command), linking each `harness/gates/<id>.yaml`.
  3. **Add an engine, step by step:**
     1. Copy `engines/robosuite/` to `engines/<id>/`.
     2. Edit the manifest (honest capabilities).
     3. Implement the six stage functions with `engines.cli_base` (contract list).
     4. Add an ExpertSim + SimBackend in `sim/<id>/`.
     5. Run the CLI dry-run tests in core and the conformance test in the engine env.
     6. Add a worker entry in `configs/workers.yaml` and start the worker with `python -m orchestrator.worker --name <id> --capabilities scene,generate,eval --port 87xx`.
     7. Add the catalog item, CREDITS, decisions.
  4. **Add an object, data method, metric or report section:** one paragraph each, naming the gate and the template.
  5. **Agent Brief:**
     - UI: the AGENT BRIEF button, or AGENT BRIEF FOR THIS STAGE in PIPELINE;
     - CLI: `python -m harness brief "..." --stage engine --engine maniskill --out brief.md`;
     - API: `POST /api/agent/brief`;
     - copy it into any agent (Copilot, Claude Code, Codex, Cursor) or RUN WITH AGENT.
  6. **Rules:** the same seven lines as `harness/brief.py` `RULES`.

- [ ] **Step 2: `engines/README.md`** (short):
  - what an engine plugin is;
  - the CLI usage block from `engines/cli_base.py`'s docstring;
  - the stage table (stage, flags, outputs), copied from the Isaac handoff §4.2 plus `--tiers`, `--max-seeds`, `--video` for eval and `--instruction`, `--max-steps` for rollout;
  - a pointer to `docs/EXTENDING.md`.

- [ ] **Step 3: Agent instruction files** (the `applyTo` frontmatter is the VS Code custom-instructions format):

`.github/instructions/engines.instructions.md`:

```markdown
---
applyTo: "engines/**,sim/**,tests/engines/**"
---
# Engine plugins

- Follow `harness/gates/engine.yaml` and `docs/EXTENDING.md`; mirror `engines/robosuite/`.
- Build every CLI with `engines.cli_base.run_cli`; import simulators lazily so `--help` works in the core env.
- Exit 0 on success, non-zero with a one-line reason on stderr; write only under `--out`.
- Core frame: robot base, table top z = 0, quaternions wxyz. The same EpisodeConfig seed is the same episode everywhere.
- Mark a capability `ready` in `engine.yaml` only after its check passes, and log the evidence in `docs/decisions.md`.
- Conformance: `python -m pytest -q tests/engines/test_conformance.py -k <engine>` in the engine's env.
```

`.github/instructions/objects.instructions.md`:

```markdown
---
applyTo: "assets/**,configs/variation/**,dataengine/variation.py"
---
# Objects and variation specs

- Never edit `configs/variation/cathlab_v1.yaml`; add `cathlab_v<N>.yaml` for new objects or axes.
- Every object needs a grasp the analytic expert understands; the expert must reach >= 80 % on it
  (`python -m engines.robosuite.cli check-expert --objects <name>`).
- Object packages live in `assets/objects/<name>/` (object.yaml, visual.obj, collision/*.obj, object.xml); importers never
  overwrite existing packages. Follow `harness/gates/objects.yaml`.
```

`.github/skills/cathlab-extend/SKILL.md`:

```markdown
---
name: cathlab-extend
description: Extend the CathLab Companion platform (new engine, object, data method, metric, report section, or any pipeline stage). Use when asked to add or change a simulator, object, synthetic data method, eval metric, rollout or report feature in this repo.
---
# Extending CathLab Companion

1. Find the stage gate in `harness/gates/` (engine, scene, objects, expert, data_method, convert, intel, train, eval,
   rollout, assist, report). Read its contract, reference implementation and context files.
2. Generate the brief: `python -m harness brief "<request>" --stage <gate> [--engine <id>]`; follow it.
3. If a harness template matches (`python -m harness list`), prefer `python -m harness run <template> --param ...`.
4. Run the gate's conformance command and `uv run pytest -q`; update status flags only for what now works.
5. Rules: AGENTS.md; human-only commit authorship (no AI Co-authored-by); no data or secrets in git; CREDITS.md and
   docs/decisions.md rows.
```

- [ ] **Step 4: Edits to the existing docs:**
  - **`AGENTS.md`:**
    - Add a section `## Extending (any agent)`: 5 bullets pointing to `docs/EXTENDING.md`, `harness/gates/`, `python -m harness brief`, the engine conformance test and `.github/skills/cathlab-extend/SKILL.md`.
    - Add the ownership row `| engines/ (SDK, robosuite, mujoco, newton) | Rakesh |`. `engines/isaac/` stays Navneet's.
  - **`docs/HARNESS.md`:**
    - Add the 5 new templates to the table, with their gates.
    - Add an "Agent Brief" section (UI, CLI, API, RUN WITH AGENT → `free_task`).
  - **`docs/INSTALL.md`:**
    - Under "Start the stack", explain that PIPELINE stages run on workers: `make worker-robosuite` for the robosuite stages and `make worker-lerobot` for convert. A stage shows *why* it's blocked.
    - Add: "A remote worker must see the same repo path and `var/` (same clone location or a shared mount), because stage commands use absolute paths under the control plane's `var/runs/`."
  - **`README.md`:** in the Control Station section, add:
    - "**PIPELINE:** pick a run, run stages (preview → check expert → generate → convert → eval → rollout) on the workers, follow logs and outputs."
    - "**AGENT BRIEF:** describe anything the UI doesn't offer; get a brief for any coding agent."
  - **`Makefile`:** add `conformance-robosuite` (`MUJOCO_GL=egl conda run -n $(ROBOSUITE_ENV) --no-capture-output python -m pytest -q tests/engines`) to `.PHONY` and as a target.
  - **Isaac handoff:**
    - §4.2: build the CLI with `engines.cli_base.run_cli` (`Flag`, `Stage`), as in `engines/robosuite/cli.py`.
    - §4.2: the orchestrator also passes `--tiers`, `--max-seeds` to `eval`, and `--instruction` to `rollout`; accept them.
    - §4.1: add `backend:` to the manifest for the conformance test.
  - **`docs/decisions.md`:** add the row `| H22.5 | Plan B-core: engine plugins (manifest + uniform CLI via engines/cli_base.py), stage pipeline dispatching engine CLIs to workers, 12 stage gates + deterministic Agent Brief | One contract for every simulator and stage, so the UI, agents and teammates extend the platform without special cases | Rakesh | [plan](plans/2026-10-23-platform-plan-b-core.md) |`.

- [ ] **Step 5: Verify:**
  - `uv run pytest -q`: all pass.
  - `make harness-list`: 12 templates.
  - `uv run python -m harness brief "Add a MuJoCo engine" --stage engine --engine mujoco | head -30`: readable.
  - `grep -rn "TODO\|TBD" docs/EXTENDING.md engines/README.md .github`: nothing.

- [ ] **Step 6: Commit**

```bash
git add docs AGENTS.md README.md Makefile engines/README.md .github
git commit -m "docs: extending guide, engine README, agent instructions and skill, Plan B-core docs updates"
```

---

## After this plan

- **B-loop (next):**
  - train stage glue (lerobot worker, `train/recipes`) for the checkpoints now on the Hub: `smolvla_panda_pick_v1`, `act_panda_pick_v1`, `xvla_panda_pick_v1`, …;
  - eval with `--policy url:` through Sharan's policy server;
  - MODELS tab (registry, promote, rollback), the ASSIST panel, the REPORT tab (Sharan's failure report → CREATE NEXT RUN).
- **B-engines:** MuJoCo (Menagerie Panda) and Newton engines against the gate above; object packages and importers; P6 data-quality rules.
