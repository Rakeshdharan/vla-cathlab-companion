# Session Handover: Sharan: Serving, Registry, CI, Dataset Intelligence, Failure Report, Docker

*From the master session, 2026-10-23 ~10:45 (≈H21). Owner: **Sharan**, on his own machine, with Rakesh supporting through the agent. Paste this file into your agent as the first message, or say "follow `docs/handoffs/2026-10-23-sharan-serve-intel-session.md`".*

## 1. Why this matters

The Control Station runs the pipeline **Scene → Generate → Intel → Train → Eval → Promote → Rollout** across simulators (robosuite, MuJoCo, Newton, Isaac). Rakesh builds the orchestrator, UI, agent harness and engines on the master PC. **You build the parts that make models servable, versioned and trustworthy.** These parts need no robotics knowledge: HTTP APIs, files, rules, data checks, CI and containers.

Your parts plug in through **fixed contracts** (§3). Build to them exactly and your work drops into the UI without further coordination.

Read first (on `main`):

| Doc | Why |
|---|---|
| [`AGENTS.md`](../../AGENTS.md) | Repo rules: lazy heavy imports, GPU-free tests, no AI co-author trailers, data never in git |
| [`docs/specs/2026-10-23-control-station-design.md`](../specs/2026-10-23-control-station-design.md) | Control plane, workers, pipeline stages |
| [`docs/specs/2026-10-23-multi-engine-platform-design.md`](../specs/2026-10-23-multi-engine-platform-design.md) | Part B: P6 data quality, P9 eval, P10 registry/promotion, P12 failure report, P16 Docker |
| `contracts/` | `PolicyAdapter`, `PolicyMeta`, `RandomPolicy`, `EvalReport`, `EpisodeResult`, `ModelCard` |
| `eval/metrics.py` | `wilson_ci`, `success_by_tier`, `success_by_object`, `latency_summary` (reuse; don't duplicate) |
| `dataengine/variation.py` | `VariationSpec`, `EpisodeConfig`, `train_configs`, `eval_configs`, `TIERS` (coverage + failure-report keys) |
| [`docs/handoffs/2026-10-23-training-gpu-session.md`](2026-10-23-training-gpu-session.md) | Training builds `train/lerobot_adapter.py: LeRobotPolicyAdapter`, which the policy server loads |

## 2. Setup

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks
git config user.name "<your name>" && git config user.email "<your email>"
uv sync && uv run pytest -q            # must be green before you start
# fastapi, uvicorn and httpx are already core dependencies on main: never `uv add` them (it would conflict).
# If you need a NEW dependency, put it in an optional group (e.g. `uv add --optional intel pyarrow`) in its own small PR.
```

For S5 you also need an env with `pyarrow`/`pandas` (or `lerobot`): `uv run --with pyarrow --with pandas ...`, or the conda `lerobot` env.

Branch per package: `feat/sharan-<topic>`. Make small PRs to `main`; each PR carries its tests.

**Stay on the tip of `main`:** start every branch from a fresh `git fetch && git switch -c feat/sharan-<topic> origin/main`, and run `git fetch && git rebase origin/main` before opening the PR. `main` moves several times a day. Never merge `main` into your branch, and never force-push `main`.

## 3. Contracts (build to these exactly)

### 3.1 Array wire format (`serve/codec.py`, stdlib + numpy only)

```python
encode_array(a: np.ndarray) -> {"b64": str, "shape": list[int], "dtype": "uint8" | "float32" | "float64" | "int64"}
decode_array(d: dict) -> np.ndarray     # exact round trip, C-contiguous
```

### 3.2 Policy server: `python -m serve.policy_server` (runs in the **lerobot** env, on a GPU box)

```bash
python -m serve.policy_server --adapter train.lerobot_adapter:LeRobotPolicyAdapter \
    --checkpoint Rakeshdharan56/smolvla-pick-robosuite --model-version cathlab-pick@v1 --io panda_libero \
    --host 127.0.0.1 --port 8720 [--token $CATHLAB_WORKER_TOKEN]
```

- `--adapter` is `module:Class` (or `module.Class`), constructed with no arguments and then given `.load(checkpoint)`. Tests use `contracts.policy:RandomPolicy`.
- The model loads in a **background thread** at startup, so `/health` answers immediately.

| Endpoint | Response |
|---|---|
| `GET /health` (no token needed) | `{"ok": true, "ready": bool, "model_version": str, "auth": bool, "error": str}` |
| `GET /meta` | `{"model_version", "io", "adapter", "checkpoint", "meta": PolicyMeta}`, or **503** `{"detail": "model loading"}` |
| `POST /act` body `{"instruction": str, "obs": {key: Array}}` | `{"actions": Array (chunk, action_dim), "model_version": str, "latency_ms": float}`; 503 while loading; 400 on a bad body; 500 `{"detail": "predict failed: ..."}` on adapter error |

- `obs` keys are whatever the simulator backend returns: robosuite/MuJoCo/Newton send `state`, `agentview`, `eye_in_hand`; Isaac sends `state`, `external_camera`, `left_wrist_camera`, `right_wrist_camera`. Pass the decoded dict straight to `adapter.predict(obs, instruction)`.
- **Auth**: if a token is set (`--token` or env `CATHLAB_WORKER_TOKEN`), every endpoint except `/health` requires header `X-Cathlab-Token`, compared with `secrets.compare_digest`; otherwise 401. Refuse to bind a non-loopback host without a token (same rule as `orchestrator/worker.py`'s `check_bind`).

### 3.3 Policy client: `serve/policy_client.py`, **stdlib + numpy only** (it runs inside the Python 3.10 robosuite env)

```python
class PolicyClient(contracts.PolicyAdapter):
    def __init__(self, url: str, token: str | None = None, timeout_s: float = 30.0, wait_ready_s: float = 60.0): ...
    def load(self, ckpt: str = "") -> None        # waits until /health says ready (≤ wait_ready_s), else RuntimeError
    def predict(self, obs: dict, instruction: str) -> np.ndarray   # float32 (chunk, action_dim); retries 503 until wait_ready_s
    def meta(self) -> PolicyMeta                  # from /meta
```

Use `urllib.request` for HTTP. The token comes from the argument or env `CATHLAB_WORKER_TOKEN`. Any eval code then does `run_suite(PolicyClient(url), backend, ...)`.

### 3.4 Registry: `serve/registry.py` + CLI `python -m serve.registry`

Storage, git-ignored under `var/`:
- `var/registry/<model_name>/<version>.yaml`: the `RegisteredModel`;
- `var/registry/<model_name>/production.yaml`: `{"current": "<version>" | null, "history": [{"version", "at", "action": "promote"|"rollback", "reason"}]}`.

```python
class RegisteredModel(BaseModel):
    card: contracts.ModelCard
    eval_reports: dict[str, str]      # engine -> path of EvalReport JSON
    registered: float

class PromotionRule(BaseModel):       # defaults from configs/serve.yaml
    primary_engine: str = "robosuite"
    latency_budget_ms_p95: float = 200.0
    min_episodes: int = 30

class PromotionDecision(BaseModel):
    promoted: bool; reason: str
    candidate_rate: float; candidate_ci: tuple[float, float]
    baseline_version: str | None; baseline_rate: float | None
    p95_ms: float

class Registry:
    def __init__(self, root: str | Path = "var/registry"): ...
    def register(self, card, eval_reports) -> RegisteredModel          # FileExistsError if the version exists
    def list_versions(self, name) -> list[RegisteredModel]             # oldest first
    def get(self, name, version) -> RegisteredModel                    # KeyError if missing
    def production(self, name) -> RegisteredModel | None
    def promote(self, name, version, rule: PromotionRule | None = None, force: bool = False) -> PromotionDecision
    def rollback(self, name) -> RegisteredModel | None                 # previous promoted version, recorded in history
```

**Promotion rule** (exactly this, from spec P10):
- The candidate needs an EvalReport for `rule.primary_engine` with ≥ `min_episodes` episodes; otherwise the result is not promoted, with reason "no/insufficient eval on <engine>".
- `p95_ms = eval.metrics.latency_summary(report)["p95_ms"]` must be ≤ the budget.
- **With no production version**, the candidate is promoted if the latency check passes.
- **Otherwise** it is promoted only if `candidate.wilson_ci()[0] > baseline.success_rate()`, where the baseline is the production version's report on the same engine.
- `force=True` skips the success check (reason "forced by user") but never the latency check.
- When promoted, set `card.promoted=True` on the candidate and `False` on the previous version, and append to the history.

CLI: `list <name>`, `show <name> <version>`, `register --card card.yaml --eval robosuite=path.json [--eval mujoco=...]`, `promote <name> <version> [--force]`, `rollback <name>`. Print JSON.

Update `configs/serve.yaml` to:

```yaml
latency_budget_ms_p95: 200.0
primary_engine: robosuite
min_episodes: 30
registry_path: var/registry
policy_server_port: 8720
```

### 3.5 Dataset intelligence: `python -m dataset_intel.report` (runs where `pyarrow` is available)

```bash
python -m dataset_intel.report <lerobot_root> [--extra <meta/episodes_extra.jsonl>] \
    [--spec configs/variation/cathlab_v1.yaml] [--io panda_libero|fr3_duo_16d] --out <dir>
```

Writes `<dir>/report.json` + `<dir>/report.md`. The JSON schema is a pydantic model `IntelReport` in `dataset_intel/model.py`:
- `dataset`: root, robot_type, fps, episodes, frames.
- `issues`: a list of `{episode, kind, detail, severity: "error"|"warning"}`. Kinds:
  - `nan` (NaN or inf in state/action);
  - `frozen` (state unchanged for ≥ 2 s outside gripper close/open holds);
  - `action_jump` (a per-step change beyond threshold);
  - `length_outlier` (IQR rule);
  - `gripper_never_closes`;
  - `idle_fraction` (no-op frames > 20 %);
  - `frame_count_mismatch` (parquet vs video frames).
- `coverage` (when `--extra` is given): counts per `(object, layout_id, obj_color_id, table_color_id, source_sim)`, `empty_cells` against `train_configs(spec)` for the spec's train grid, and `per_object` counts.
- `readiness`: `score` in [0, 1] plus its components (valid fraction, coverage fraction, object balance, idle penalty). Weights and thresholds live in `configs/intel.yaml`.

**I/O profile** sets the gripper convention:
- `panda_libero`: closed when `action[6] > 0`;
- `fr3_duo_16d`: right gripper `action[15] < 0.5`, where 1 = open.

Must run on `Rakeshdharan56/pick_robosuite` (public, revision `v1`) and on the provided `Rakeshdharan56/fr3_duo_handover_84ep` (private; no `--extra`, so coverage is skipped). Paste both summaries in the PR.

### 3.6 Failure report: `python -m report.failure_report`

```bash
python -m report.failure_report --eval robosuite=<eval_report.json> [--eval mujoco=...] \
    --spec configs/variation/cathlab_v1.yaml --model cathlab-pick@v1 --out <dir> [--top 10] [--k 8]
```

- **Rebuild each episode's cell:** for an `EpisodeResult` with `scene_variant` = tier and `seed`, take `eval_configs(spec, tier, [seed])` and the entry whose `object` matches. That gives `(engine, tier, object, layout_id, obj_color_id, table_color_id)`.
- **Report contents:**
  - success per engine, tier and object, with Wilson CIs (via `eval.metrics`);
  - failure-stage histogram (`EvalReport.stage_failure_hist()`);
  - wrong-object count (from `judge_verdict` when present);
  - sim2sim gap: primary engine minus each other engine, per tier;
  - top-`N` failing cells;
  - example failure videos (`video_path`).
- **Recommendations** per failing cell, by rule:
  - fails at `reach`/`grasp` → "generate K targeted episodes";
  - fails at `lift`/`transport` → "check expert params / friction range";
  - fails only on held-out colour or table → "widen colour palette in next spec version".
- **Next run:** `next_run.json` = `{"spec": ..., "cells": [{object, layout_id, obj_color_id, table_color_id, k}]}`. Rakesh's Generate stage consumes it.
- Outputs `report.json`, `report.md` and `next_run.json`.

### 3.7 Docker (`docker/`, `compose.yaml`)

The images:
- `docker/control.Dockerfile`: python:3.11-slim + uv; `uv sync --frozen`; runs `python -m orchestrator.control --host 0.0.0.0 --port 8700 --no-agent`.
- `docker/frontend.Dockerfile`: nginx serving `frontend/`, proxying `/api` to `control:8700`.
- `docker/worker-robosuite.Dockerfile`: python:3.10; robosuite 1.5.2, mujoco 3.3.7, numpy<2, pydantic, pyyaml, imageio, fastapi, uvicorn, httpx; `MUJOCO_GL=egl` (libegl1, libgl1); runs `python -m orchestrator.worker --name robosuite --capabilities preview,check-expert,generate,eval,rollout --host 0.0.0.0 --port 8711`.
- `docker/worker-lerobot.Dockerfile`: CUDA 12 base; `lerobot>=0.5,<0.6`, fastapi, uvicorn, httpx; worker on :8712 (`convert,intel,train,policy_server`).

Plus:
- `compose.yaml` with these four services, a shared `var/` volume, `CATHLAB_WORKER_TOKEN` from `.env`, and GPU reservation for the lerobot worker;
- `configs/workers.docker.yaml` (service-name URLs), selected with `CATHLAB_WORKERS=configs/workers.docker.yaml`;
- a "Docker" section in `docs/INSTALL.md`: build, up, logs, GPU prerequisites (NVIDIA Container Toolkit), and why the agent harness and Isaac stay on their hosts.

`orchestrator/` (worker, control plane) is on `main`, and `frontend/` and `orchestrator/control.py` land later today (Plan A). Start S7 once `orchestrator/control.py` exists on `main`.

## 4. Work packages (priority order; each ends with a ✅)

| # | Package | Files | ✅ Done when |
|---|---|---|---|
| **S1** | Codec + policy server | `serve/__init__.py`, `serve/codec.py`, `serve/policy_server.py`, `tests/serve/test_codec.py`, `tests/serve/test_policy_server.py` | Tests (FastAPI TestClient, `RandomPolicy` adapter): health before/after load, 503 while loading, `/act` shape `(chunk, dim)`, 400 on a bad body, token 401/200, `check_bind` refusal |
| **S2** | Policy client | `serve/policy_client.py`, `tests/serve/test_policy_client.py` | A real uvicorn server on a free port in a thread + `PolicyClient`: predict shape and dtype, waits through 503, token passthrough; `eval.runner.run_suite(PolicyClient(url), DummySim(), ...)` returns an EvalReport |
| **S3** | Registry + promotion + rollback + CLI | `serve/registry.py`, `configs/serve.yaml`, `tests/serve/test_registry.py` | Tests: first promote; candidate beats baseline (promoted); candidate within the CI (rejected, with reason); latency over budget (rejected, even with force); insufficient episodes; rollback returns the previous version and history records it; duplicate register raises |
| **S4** | CI | `.github/workflows/ci.yml` | On push/PR to `main`: ubuntu-latest, uv, Python 3.11, `uv sync`, `uv run pytest -q`. Green on your PR |
| **S5** | Dataset intelligence | `dataset_intel/{__init__,model,validators,coverage,report}.py`, `configs/intel.yaml`, `tests/dataset_intel/` (`pytest.importorskip("pyarrow")`; a tiny synthetic parquet fixture) | Unit tests for each validator kind and the readiness score; real runs on `pick_robosuite@v1` and `fr3_duo_handover_84ep` pasted in the PR |
| **S6** | Failure report (stretch) | `report/{__init__,failure_report}.py`, `tests/report/` | Tests with a hand-made EvalReport: cell reconstruction, stage histogram, gap, recommendations, `next_run.json` |
| **S7** | Docker + compose | `docker/*.Dockerfile`, `compose.yaml`, `configs/workers.docker.yaml`, `docs/INSTALL.md` (Docker section) | `docker compose build`, then `docker compose up control frontend worker-robosuite`: the UI on :8080 shows the ROBOSUITE lamp green |

After S1–S3, post the short report (§5). Rakesh wires the Eval/Promote/Rollout stages to your server, client and registry.

## 5. Report back to the master session (after S3, S5, S7)

Post the PR links, test counts, the server/client/registry CLI examples, the S5 summaries for both datasets, and any contract deviation. No contract changes without asking first.

## 6. Rules

- **Only touch** `serve/`, `dataset_intel/`, `report/`, `docker/`, `compose.yaml`, `.github/workflows/`, `configs/serve.yaml`, `configs/intel.yaml`, `configs/workers.docker.yaml`, `tests/{serve,dataset_intel,report}/`, `docs/INSTALL.md` (Docker section) and `docs/decisions.md` (rows).
- **Never touch** `contracts/`, `dataengine/`, `orchestrator/`, `harness/`, `frontend/`, `sim/` or `engines/`; ask Rakesh instead.
- **Tests:** `uv run pytest -q` stays GPU-, simulator- and network-free. Import heavy packages (`torch`, `lerobot`, `pyarrow`) inside functions, or skip with `pytest.importorskip`.
- **Never commit** data, checkpoints, `.env` or tokens. Restricted Siemens data stays private on HF and only ever gets analysed, never re-uploaded publicly.
- **Commits** carry your name only; the hook strips AI `Co-authored-by` trailers.
