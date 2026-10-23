# CathLab Control Station: Orchestration, UI, Agent Harness, Eval & Rollout (Design Spec)

*Approved 2026-10-23 ~07:00 (≈H17). Builds on [the learning-loop spec](2026-10-22-cathlab-learning-loop-design.md) and [the simulation framework spec](2026-10-22-sim-framework-design.md). Owner: Rakesh.*

## 1. Goal

Turn the repo into a **Control Station**: one industrial UI plus a backend that lets anyone:

1. pick the project inputs (simulator, embodiment, objects, expert, variation, model) from a **catalog**;
2. run the end-to-end pipeline **Scene → Generate → Intel → Train → Eval → Promote → Rollout** with live progress and outputs;
3. turn any missing or extendable catalog item into a reviewed git branch through the **agent harness** ("Build with agent").

**Must work:**
- natively today (conda envs + one command);
- in Docker later, with the same code and the same ports.

## 2. Decisions

| # | Decision |
|---|---|
| C1 | **Topology:** one codebase with two roles. A **control plane** (FastAPI) and **workers**: the same small worker app started inside each Python environment. The control plane dispatches stages to workers by capability |
| C2 | **Workers:** `robosuite` (py3.10: generate, eval-rollout, scene-preview), `lerobot` (py3.12, GPU: convert, intel, train, policy-server), `isaac` (GPU machine, later: generate, eval-rollout), `agent` (**always on the host**: harness). The cloud GPU box runs a `lerobot` worker registered by URL |
| C3 | **Eval and rollout go through a policy server.** A LeRobot worker serves a registered model version over HTTP. The robosuite worker runs `RobosuiteBackend` with a `PolicyClient`. Needed because the robosuite env can't import LeRobot; the same path later serves a real robot |
| C4 | **Catalog** (`configs/catalog.yaml`) lists every option with status `ready` / `in_progress` / `planned` and an optional `harness_task`. A **Run** = a frozen selection snapshot `runs/<id>/run.yaml` |
| C5 | **Agent harness:** typed task templates (`harness/tasks/*.yaml`). Each job runs in its own git worktree on `agent/<task>-<id>` from `origin/main`. Agent CLI is pluggable (`configs/agents.yaml`), Copilot CLI by default (`copilot -C <wt> -p … --allow-all-tools --output-format json`), Claude Code / Codex optional. Then a path guard, acceptance commands, the diff, and **human review → Push branch / Discard**. Also usable from the CLI: `python -m harness run <task> --param k=v` |
| C6 | **Frontend:** static HTML/CSS/ES-module JS, no Node build. Layout = PolyScope-style header tabs (PROJECT, PIPELINE, DATA, MODELS, AGENT, LOG), project tree, Run/Stop footer with the live job, and the 7-stage strip on the PIPELINE tab. Industrial style: grey panels, navy bars, one blue accent, status lamps, monospace values, no gradients or illustrations |
| C7 | **State:** SQLite (`var/control.db`) for jobs, runs and the registry; artifacts on disk under `var/` (git-ignored); logs streamed with Server-Sent Events |
| C8 | **Docker:** Dockerfiles and `compose.yaml` written alongside, validated at the end. Services: `frontend` (nginx), `control`, `worker-robosuite`, `worker-lerobot` (GPU). The agent worker stays on the host |
| C9 | **Baseline model = SmolVLA** (Akanksh's lock-in). ACT is the optional second model |
| C10 | **Step 0 integration:** merge `feat/rakesh-robosuite-base` into `main`, restore the real `contracts/__init__.py` exports (Mitesh's stub types in `contracts/types.py` shadow them), add `"robosuite"` to `Backend`, and make `uv run pytest` green |

## 3. Components

| Path | Responsibility |
|---|---|
| `orchestrator/catalog.py` | Load and validate `configs/catalog.yaml`; selection → `RunConfig`; status rules |
| `orchestrator/store.py` | SQLite: `jobs`, `runs`, `models` tables; small typed functions, no ORM |
| `orchestrator/stages.py` | Stage registry: name, required capability, command builder (`RunConfig` → argv + env), outputs collector, dependencies |
| `orchestrator/worker.py` | Worker app: `GET /health` (name, capabilities, env), `POST /jobs` (run argv, stream log), `GET /jobs/{id}`, `GET /jobs/{id}/log` (SSE), `POST /jobs/{id}/cancel` |
| `orchestrator/control.py` | Control app: `/api/catalog`, `/api/runs`, `/api/runs/{id}/stages/{stage}` (start), `/api/jobs`, `/api/jobs/{id}/log` (SSE proxy), `/api/workers`, `/api/models`, `/api/models/{v}/promote`, `/api/rollout`, `/api/agent/*`; serves `frontend/` in dev |
| `orchestrator/workers.yaml` | Worker registry: name → URL (dev defaults `localhost:87xx`; Docker service names) |
| `harness/` | Task model, prompt renderer, worktree manager, agent runner, path guard, acceptance runner, CLI `python -m harness` |
| `harness/tasks/*.yaml` | `add_object`, `extend_dataset`, `add_variation_axis`, `add_sim_backend`, `add_policy_adapter`, `add_expert`, `free_task` |
| `serve/policy_server.py` | HTTP policy server (lerobot env): load a LeRobot checkpoint, `POST /act` {obs, instruction} → action chunk; `GET /meta` |
| `serve/policy_client.py` | `PolicyClient(contracts.PolicyAdapter)` over HTTP (stdlib only, so it runs in py3.10) |
| `serve/registry.py` | ModelCard registration, promotion rule (CI beat + p95 latency ≤ `configs/serve.yaml` budget), production pointer, rollback |
| `dataset_intel/` | Minimal validators + coverage over the variation grid + readiness report (Sharan's work can replace it) |
| `scripts/stage_*.py` | Thin stage entry points that workers execute (scene preview, generate shard, convert, intel, train, eval, rollout) |
| `frontend/` | `index.html`, `css/station.css`, `js/*.js` (api client, tabs, project tree, stage strip, agent console, log viewer) |
| `docker/`, `compose.yaml` | Images and compose |
| `docs/INSTALL.md`, `docs/HARNESS.md`, `CREDITS.md` | Install (native + Docker), harness guide for teammates, credits |

## 4. Pipeline stages

| Stage | Worker | Command (wrapped by `scripts/stage_*.py`) | Outputs |
|---|---|---|---|
| scene | robosuite | `robosuite_variation_grid.py` for N configs | `var/runs/<id>/scene/grid.png` |
| generate | robosuite (k shards) → lerobot (convert) | `robosuite_generate.py --shard i --num-shards k` → `npz_to_lerobot.py`; optional HF push | dataset dir + `manifest.yaml`; success per object; episodes/min; preview GIF |
| intel | lerobot | `python -m dataset_intel.report <dataset>` | `report.json` + `report.md`: issues, coverage grid, readiness score |
| train | lerobot (GPU) | `lerobot-train --policy.type=smolvla …`, then `ModelCard` | checkpoint dir, ModelCard, loss log |
| eval | robosuite + lerobot policy-server | start the server with checkpoint X; robosuite worker runs `run_suite` over suite v2 with `PolicyClient`; also the expert as a reference | `EvalReport` JSON, metrics summary |
| promote | control | registry rule vs the current production version | decision + reason |
| rollout | lerobot policy-server + robosuite | serve version X; run 1 live episode for the chosen object, recording video | mp4, served version, latency |

**Eval suite v2:** `configs/eval/suite_v2.yaml`: objects bottle/vial/syringe × tiers (`dataengine.variation.TIERS`) × seeds 5000–5003 = 60 episodes per model per simulator.

## 5. Agent harness contract

Task template fields:
- `id`, `title`, `description`
- `catalog_target`
- `params` (name, type, default, choices, help)
- `context` (files)
- `prompt` (Jinja-free `str.format` template)
- `allowed_paths` (globs)
- `acceptance` (list of `{name, cmd, cwd}`)
- `timeout_min`

Job states: `created → worktree → agent_running → guard → checks → ready_for_review → pushed | discarded | failed`.

Artifacts per job in `var/agent/<id>/`: `prompt.md`, `agent.jsonl`, `checks.log`, `diff.patch`, `result.json`.

The agent never runs against `main` or the user's working folder. Commits use the invoking user's git identity; the repo hook strips AI trailers.

## 6. Error handling

- Every job has a state and a full log. A failed stage blocks the stages that depend on it, and the UI shows the log tail and the reason.
- Worker health is polled every 5 s. Stages whose capability has no healthy worker are disabled, with the reason.
- The harness has a timeout, a path-guard failure → `failed` with the offending paths, and a check failure → `failed` with the logs (the worktree is kept for inspection until discarded).

## 7. Testing

- Unit tests run GPU- and simulator-free:
  - catalog, store, stage command builders;
  - the worker app with an `echo`-style job;
  - control routes (FastAPI TestClient);
  - harness with a fake agent command (a script that edits a file) on a temporary git repo;
  - registry rule;
  - policy server/client with a dummy policy.
- Sim/GPU tests are marked or skipped if the dependencies are missing.
- End-to-end smoke (`make smoke`): control + robosuite worker + lerobot worker natively, with a tiny run (3 episodes, a 50-step train, a 3-episode eval).

## 8. Credits

Credited in `CREDITS.md` and the README:
- robosuite
- MuJoCo
- LeRobot
- SmolVLA
- ACT
- NVIDIA Isaac Sim / Isaac Lab
- Hugging Face Hub
- FastAPI / uvicorn
- pydantic
- GitHub Copilot CLI
- the LIBERO conventions
- *A tutorial note on collecting simulated data for VLA models* (arXiv 2508.06547)
