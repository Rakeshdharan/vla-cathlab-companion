# AGENTS.md — CathLab Companion

Siemens Healthineers Hackathon 2026 (C2). **Current sim design:** single-arm Franka Panda, own table scene in robosuite + ManiSkill3 (engine plugins under `engines/`; Isaac Sim was dropped at H24), LIBERO I/O, task "Hand me the {bottle | vial | syringe}". See [docs/specs/2026-10-22-sim-framework-design.md](docs/specs/2026-10-22-sim-framework-design.md). We build a **simulation-first, self-improving robot learning loop**: a Franka Panda hands cathlab items to a clinician ("Hand me the syringe"). 36 h, 5 people, each working with coding agents in this one repo.

Read these before non-trivial work:
- Design spec: [docs/specs/2026-10-22-cathlab-learning-loop-design.md](docs/specs/2026-10-22-cathlab-learning-loop-design.md)
- Foundation plan: [docs/plans/2026-10-22-cathlab-loop-implementation.md](docs/plans/2026-10-22-cathlab-loop-implementation.md)
- Workstreams and owners: [docs/tasks/README.md](docs/tasks/README.md) (one file per person)
- Decisions log: [docs/decisions.md](docs/decisions.md)
- Provided datasets, HF repos and schema: [docs/data.md](docs/data.md)

## Architecture

Pipeline: `01 ingest → 02 dataset_intel → 03 train → 04 eval → 05 serve`, closed by `loop/` (failures → gap → synthetic data → retrain → promote).

- `contracts/` is the **only** coupling between packages:
  - `TaskSpec`, `SceneSpec`, `DatasetManifest`;
  - `PolicyAdapter` (`load` / `predict(obs, instruction) -> (chunk, action_dim)` / `meta`);
  - `SimBackend` (`reset` / `step` / `render` / `close`);
  - `EvalReport`, `ModelCard`.
- Every model is a `PolicyAdapter` and every simulator is a `SimBackend`. Never import another workstream's internals; import from `contracts` or that package's public `__init__`.
- Stages exchange **versioned artifacts**, not in-memory objects: LeRobot datasets + `manifest.yaml`, `EvalReport` JSON, `ModelCard` YAML.
- `DummySim` and `RandomPolicy` exist so every package can be built and tested without a simulator or a GPU.

| Package | Owner |
|---|---|
| `contracts/`, `dataengine/` (shared sim core), `ingest/`, `train/`, `loop/`, `sim/robosuite/`, `sim/robosimstudio/` | Rakesh |
| `dataset_intel/`, `serve/`, `report/`, `docker/`, `compose.yaml`, `.github/workflows/`, `configs/{serve,intel,workers.docker}.yaml`, `docs/research/` | Sharan |
| `awareness/` | Navneet |
| `sim/maniskill/`, `engines/maniskill/` | Mitesh (Rakesh built v1) |
| `sim/genesis/`, `engines/genesis/` | Akanksh |
| `assets/` | Rakesh |
| `eval/` (beyond `runner.py` signatures), `configs/eval/` | Akanksh |
| `orchestrator/`, `harness/`, `frontend/`, `configs/{catalog,workers,agents}.yaml` | Rakesh |

## Build and test

```bash
uv sync                 # install core + dev deps
uv run pytest           # must stay green on main; tests must not need GPU/simulators/network
uv sync --extra train   # LeRobot + W&B (training machines only)
```

Heavy dependencies (simulators, LeRobot, GraspGen, cuRobo, ElevenLabs) are optional extras or lazy imports. Never import them at module top level in code that the tests reach.

## Conventions

- **TDD:** write the failing test first, using `DummySim` / `RandomPolicy` / tiny fixtures.
- **Contracts are frozen.** A change to `contracts/` needs a dedicated PR reviewed by Rakesh, and must update every implementer in the same PR.
- **Stay in your lane:** edit only the packages of the workstream you're working on. If you need something from another package, write the request in the PR description or in `docs/decisions.md`.
- Task stages are exactly `reach, grasp, lift, transport, place`. Dataset `source` is one of `provided, domain_rand, mimicgen, scripted, teleop, cosmos`.
- Config lives in `configs/` (YAML, validated by pydantic models). Don't hard-code paths, object lists or seeds.
- Large artifacts (datasets, checkpoints, videos, USD assets > 10 MB) never go into git. Use HF Hub / W&B Artifacts and reference them by URI in manifests/model cards.
- Secrets (HF, W&B, ElevenLabs, VLM API keys) come from a git-ignored `.env`. Never print, log or commit keys.
- W&B project `cathlab-companion`; run name `<model>-<dataset_version>-<owner>`.
- Record every non-obvious choice as one line in `docs/decisions.md`. These feed the pitch.
- Only comment code that needs clarification. Each package has a short `README.md`.

## Git and commits

- Branch: `feat/<owner>-<topic>`; small PRs to `main`; rebase before merge.
- **`main` is the source of truth and moves often (Rakesh drives it).** Always branch from the tip: `git fetch && git switch -c feat/<owner>-<topic> origin/main`. Before opening or updating a PR, `git fetch && git rebase origin/main` and re-run `uv run pytest -q`. Never merge `main` into a feature branch. Don't add dependencies that already exist on `main` (check `pyproject.toml`); new ones go in an optional group, in their own small PR.
- Commit messages: Conventional Commits (`feat(eval): …`, `fix(serve): …`, `docs: …`).
- **Commits are authored by the human teammate only.** Do NOT add `Co-authored-by:` trailers for AI agents/assistants (Copilot, Claude, Codex, Cursor, etc.) and do not sign commits as an agent. The `.githooks/commit-msg` hook strips such trailers; enable it once per clone with `git config core.hooksPath .githooks`.
- Never force-push `main`, and never commit `.env`, datasets, or checkpoints.

## Control Station and agent harness

- Control plane: `python -m orchestrator.control` (FastAPI, port 8700) serves `frontend/`, the catalog (`configs/catalog.yaml`), runs/jobs (`var/control.db`) and `/api/agent/*`.
- Workers: `python -m orchestrator.worker --name <env> --capabilities ... --port 87xx` inside each Python env; URLs in `configs/workers.yaml`.
- Harness: task templates in `harness/tasks/*.yaml` (file name = id). To support a new kind of agent task, add a template rather than special-casing code. Agents run in worktrees on `agent/<task>-<id>`, may only change `allowed_paths`, must not commit or push, and every job ends with a human review.
- When you add a catalog item that is not ready, set `harness_task` to an existing template id (`tests/harness/test_templates.py` enforces this).
- **Always stay on the tip of `main` when running the control plane and starting agent jobs.** New templates, catalog items, and configuration changes are published frequently.
