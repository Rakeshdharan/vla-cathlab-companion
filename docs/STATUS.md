# Project status: data, models, engines, codebase

*Single source of truth for what exists on HF and what is next. Last checked against HF and `main`: **2026-10-23 17:50 (≈ H27)**; codebase section at `main` 98303b7. Update this file when a dataset or checkpoint lands.*

> **Model set (decided H26.5): ACT, SmolVLA, X-VLA, with the scripted expert as the baseline**, for both Panda and FR3. π0.5 and MolmoAct2 are dropped.

## Datasets (LeRobot v3)

| Dataset | Robot / I/O | Engine (physics) | Episodes | Frames | Status | Notes |
|---|---|---|---|---|---|---|
| [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) `v1` | Panda, `panda_libero` (7-D EEF delta, 8-D state, 2 cams 256², 20 fps) | robosuite 1.5.2 (MuJoCo 3.3.7) | 384 | 69,191 | ✅ public | Full train grid, expert 100 %. Raw shards: `pick_robosuite_raw` |
| [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) `v1` | Panda, `panda_libero` | ManiSkill3 3.0.1 (**PhysX 5**) | 369 / 384 | 45,387 | ✅ public | Paired with robosuite episode for episode; expert bottle 100 / vial 90 / syringe 100 %. Raw: `pick_maniskill_raw`. [PR #10](https://github.com/Rakeshdharan/vla-cathlab-companion/pull/10) |
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) `v1`, **`v1.0.1`** | FR3 Duo, `fr3_duo_16d` (16-D joint targets, 3 cams 224², 30 fps), right arm | Newton 1.6 (MuJoCo-Warp 3.12) | 384 | 159,568 | ✅ public | Expert 128/128 per object. **Use `v1.0.1` for π0.5 / QUANTILES models**: it widens q01/q99 of the idle left-arm columns (otherwise normalised to ±30). Same episodes as `v1`. [Report](https://github.com/aravinds1610/mujoco_cathlab_vla/blob/main/docs/NEWTON_REPORT.md) |
| Newton "more episodes" (seeds 1, 2) | FR3 Duo, 16-D | Newton | 768 generated | — | ⏸ **not published** | Resampled physics/light + ±1 cm layout; 768/768 success. Publish as `v1.1` if FR3 models need more data |
| `pick_genesis` | Panda | Genesis 1.4.5 | — | — | ⬜ not started | Handoff ready; optional (PhysX already gives cross-physics) |

## Model checkpoints

| Model | Panda, robosuite data (`pick_robosuite@v1`) | Panda, ManiSkill data (`pick_maniskill@v1`) | FR3 Duo, Newton data (`cathlab_fr3_newton_v1@v1.0.1`) |
|---|---|---|---|
| Scripted expert (baseline) | ✅ robosuite | ✅ ManiSkill 60/60 on suite_v2 | ✅ Newton + MuJoCo 100 % on every tier |
| ACT (from scratch) | ✅ [`act_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/act_panda_pick_v1), 50,000 × 64 (baseline) | dropped | dropped |
| SmolVLA | ✅ [`smolvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_pick_v1), 8,500 × 32, from `smolvla_libero` | ⏳ `Rakeshdharan56/smolvla_panda_maniskill_v1` (RTX 6000 Ada) | ⏳ `aravindsuresh16/smolvla_fr3_pick_v1`, from `smolvla_base`, **Aravind, cloud GPU, ~20:30** |
| X-VLA | ✅ [`xvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_pick_v1), 7,500 × 32, from `xvla-base` (`auto`) | ⏳ `Rakeshdharan56/xvla_panda_maniskill_v1` (RTX 6000 Ada) | ⏳ `Rakeshdharan56/xvla_fr3_pick_v1`, from `xvla-base` (`auto`, 16-D, 3 cams), **RTX 6000 Ada, ~21:30** |
| ~~π0.5~~, ~~MolmoAct2~~ | dropped | — | dropped |
| **Panda v2** (co-training) | ⬜ optional: `pick_panda_v2` = merged robosuite + ManiSkill (753 eps) → `smolvla_panda_pick_v2` | | — |

**ACT is dropped for the new datasets** (it stays as the robosuite-trained baseline). FR3 checkpoints are evaluated by Aravind in Newton + MuJoCo. The ManiSkill runs use **exactly** the v1 recipes (only the dataset and repo ids differ, proven by a `train_config.json` diff): [`handoffs/2026-10-23-maniskill-training-session.md`](handoffs/2026-10-23-maniskill-training-session.md). Add VRAM, steps/s and ETAs here as the runs start.

**Cross-physics eval matrix (the headline, suite_v2, 60 episodes per cell):**

| | eval robosuite (MuJoCo) | eval ManiSkill (PhysX) |
|---|---|---|
| expert | ⬜ | ✅ 60/60 |
| ACT / SmolVLA / X-VLA trained on robosuite | ⬜ | ⬜ |
| SmolVLA / X-VLA trained on ManiSkill | ⬜ | ⬜ |

Training details: [`handoffs/2026-10-23-training-gpu-session.md`](handoffs/2026-10-23-training-gpu-session.md).

**Conventions for new checkpoints** (so eval and the registry find them):
- repo `<user>/<model>_<embodiment>_pick_v<N>`, e.g. `aravindsuresh16/smolvla_fr3_pick_v1`;
- `train_config.json` kept: it records the dataset `repo_id@revision`;
- the model card names the dataset version.

## Engines

| Engine | Robot | Physics | Plugin on `main` | Data | Eval |
|---|---|---|---|---|---|
| robosuite | Panda | MuJoCo 3.3.7 | ✅ `engines/robosuite` | ✅ | ✅ expert; checkpoints blocked (see below) |
| ManiSkill3 | Panda | PhysX 5 | ✅ `engines/maniskill` | ✅ | ✅ expert 60/60; checkpoints blocked |
| Newton | FR3 Duo | MuJoCo-Warp | ⬜ Aravind's repo only ([`cathlab_sim/newton`](https://github.com/aravinds1610/mujoco_cathlab_vla/tree/main/cathlab_sim/newton)) | ✅ | expert 100 % all tiers, sim2sim vs MuJoCo 60/60 |
| Genesis | Panda | Genesis 1.4.5 | 🟡 branch `feat/navneet-genesis-engine` (Navneet: plugin, tests, README, decisions), **PR + review pending** | ⬜ | ⬜ |
| ~~Isaac Sim~~ | ~~FR3 Duo~~ | ~~PhysX~~ | removed (H24) | — | — |

## Codebase readiness (`main` 98303b7, H27)

`uv run pytest -q` → **242 passed, 10 skipped** (GPU- and simulator-free). Simulator tests run in each engine's env (`make conformance-robosuite`, `make conformance-maniskill`).

**Legend:** ✅ works end to end · 🟡 partial · ⬜ missing.

| Area | Code | State | What works | What's missing |
|---|---|---|---|---|
| Contracts | `contracts/` | ✅ | TaskSpec, SceneSpec (backends: dummy, robosuite, maniskill, genesis, mujoco, newton, robosimstudio), DatasetManifest, PolicyAdapter, SimBackend, EvalReport (Wilson CI), ModelCard, GapSpec | — |
| Shared sim core | `dataengine/` | ✅ | Variation spec + paired `EpisodeConfig` grid, 5 eval tiers, waypoint expert, fake sim, npz sink, LeRobot sink, generator registry | P6 data-quality rules (no-op drop, success hold) not in the shared sink yet |
| Engine SDK | `engines/{registry,cli_base,policies,expert_policy}.py` | ✅ | Manifests + I/O profiles (`panda_libero`, `fr3_duo_16d`), uniform CLI (6 stages, `--run`, `--dry-run`, one-line errors), expert-as-policy, conformance test | `url:` policies wait for `serve/policy_client.py`; `lerobot:<ckpt>` adapter is being built in the RTX 6000 ManiSkill session |
| robosuite engine | `engines/robosuite`, `sim/robosuite` | ✅ | preview, check-expert, generate, convert, eval (suite_v2), rollout (GIF; no `imageio_ffmpeg` in the env) | trained-checkpoint eval (adapter) |
| ManiSkill3 engine | `engines/maniskill`, `sim/maniskill` | ✅ | All 6 stages; expert 60/60 on suite_v2; worker :8714 (`make worker-maniskill`) | Env only on the RTX 6000 Ada (no `maniskill` env on this box) |
| Genesis engine | branch `feat/navneet-genesis-engine` | 🟡 | Plugin + tests on the branch | PR, review, merge, dataset |
| Newton / MuJoCo FR3 | Aravind's repo | 🟡 | Data on HF, expert 100 %, sim2sim 60/60 | Not an engine plugin in this repo (optional) |
| Control plane + workers | `orchestrator/` | ✅ | Catalog + selection rules, runs (`var/runs/<id>/run.yaml`), SQLite jobs, worker app (token auth, Host allow-list), **stage pipeline**: readiness with reasons, dispatch to workers, logs, cancel, outputs, no double dispatch | Train / version / promote stages not in `orchestrator/stages.py` yet |
| Agent harness | `harness/` | ✅ | 12 templates, worktree jobs with path guard + checks + human review, env allow-list, 12 **stage gates**, deterministic **Agent Brief** (UI, CLI `python -m harness brief`, API) | Frozen-file deny-list (e.g. `cathlab_v1.yaml`) in the guard |
| UI | `frontend/` | 🟡 | PROJECT (catalog, Build with agent), **PIPELINE** (6 stages, live log, outputs, video), AGENT, LOG, **AGENT BRIEF** drawer | DATA and MODELS tabs are placeholders; no ASSIST or REPORT tab |
| Eval metrics + judge | `eval/` | 🟡 | metrics, failure classes, VLM judge, `eval.cli` (DummySim/robosimstudio), tests | Benchmark table across models × engines; engine evals feed it once checkpoints can be loaded |
| Awareness | `awareness/` | 🟡 | Procedure model, VLM step tracking, propose-confirm assistant (tests) | ASSIST panel wiring (instruction → rollout), voice |
| Training | `train/` | ⬜ | Recipes and checkpoints exist on the RTX 6000 / HF | `train/lerobot_adapter.py` (**the eval blocker**), recipes on `main`, Train stage in the UI |
| Serving + registry | `serve/` | ⬜ | Contract in the [Sharan handoff](handoffs/2026-10-23-sharan-serve-intel-session.md) | Policy server/client, registry with promote/rollback; **now owned by Rakesh** (Sharan is out) |
| Dataset intelligence | `dataset_intel/` | ⬜ | Contract (S5) | Validators, coverage, readiness score; Rakesh |
| Failure report → next run | `report/` | ⬜ | Contract (S6) | `report.md/json` + `next_run.json` → CREATE NEXT RUN; Rakesh |
| Loop controller | `loop/` | ⬜ | `contracts.GapSpec` | Gaps → targeted generate → v2 |
| Ingest | `ingest/` | ⬜ | Provided FR3 data on HF (private) | Validation report (low priority) |
| CI + Docker | `.github/workflows/`, `docker/`, `compose.yaml` | ⬜ | — | Proposed for Mitesh (free after ManiSkill was merged) |
| Docs + agent setup | `AGENTS.md`, `docs/EXTENDING.md`, `docs/HARNESS.md`, `docs/INSTALL.md`, `.github/{instructions,skills,prompts,agents}` | ✅ | Install, harness guide, extending guide (robosuite + ManiSkill examples), engine README, per-area agent instructions, repo skill | — |

**Critical path to the demo:**
1. `train/lerobot_adapter.py` on `main`.
2. Eval ACT / SmolVLA / X-VLA in robosuite + ManiSkill from PIPELINE.
3. Registry + promote/rollback (MODELS tab).
4. Rollout of the production model with video.
5. Failure report → `next_run.json` → CREATE NEXT RUN.
6. ASSIST panel.

## The loop after training

Every stage below already has a contract (spec Part B P9–P12 and Sharan's [handoff](handoffs/2026-10-23-sharan-serve-intel-session.md)). **What's missing is the code on `main`**:

| Stage | What it produces | Owner | Code on `main` | State |
|---|---|---|---|---|
| **Eval** | `EvalReport` per model × engine × tier (suite_v2: 3 objects × 5 tiers × seeds 5000–5003 = 60 episodes); expert as reference | Rakesh (Panda), Aravind (FR3) | ✅ `engines/{robosuite,maniskill}/cli eval` (expert works); **needs a policy adapter** | ⛔ blocked on `train/lerobot_adapter.py` or `serve/policy_client.py` |
| **Version** | Registry entry `cathlab-pick@vN` = ModelCard (checkpoint URI, dataset versions, eval report, W&B run, git SHA) | Rakesh (was Sharan, S3) | ⬜ `serve/registry.py` | not started on `main` |
| **Promote / rollback** | Production pointer moves only if the candidate beats production beyond the 95 % CI and p95 latency ≤ 200 ms (`configs/serve.yaml`) | Rakesh (was Sharan, S3) | ⬜ | — |
| **Deploy** | Policy server serving the production version (HTTP+JSON, token auth) + stdlib `PolicyClient` used by every engine | Rakesh (was Sharan, S1, S2) | ⬜ `serve/policy_server.py`, `serve/policy_client.py` | — |
| **Rollout** | One live episode of the production model with video, from the Control Station (ASSIST/PIPELINE) | Rakesh | ✅ `engines/<id>/cli rollout` (expert) | waits for Deploy |
| **Failure report → next loop** | Failing cells (engine × tier × object × layout/colour), stage histogram, wrong-object grasps, sim2sim gap, `next_run.json` with exactly those cells → generate → v2 data → retrain | Rakesh (was Sharan, S6) | ⬜ `report/failure_report.py` | — |

## Blocker for every headline number

**No trained checkpoint has been evaluated closed-loop yet.** Engine `eval --policy url:<server>` needs the policy server (`serve/`, now Rakesh), or the in-process `train/lerobot_adapter.py` from the training session. **Neither is on `main`.** Whichever lands first unblocks the eval matrix: 5 Panda checkpoints × {robosuite, ManiSkill} × suite_v2 tiers, plus FR3 checkpoints in Newton.

## Next (owner)

1. **RTX 6000 Ada:** SmolVLA + X-VLA on `pick_maniskill@v1` and X-VLA on the FR3 Newton data; the `lerobot:<ckpt>` policy adapter, and all 6 checkpoints + the expert evaluated in robosuite and ManiSkill. **Rakesh** ([handoff](handoffs/2026-10-23-maniskill-training-session.md)).
2. **Serving + registry:** policy server/client (S1–S2), registry with promote/rollback (S3), failure report (S6). **Rakesh** (Sharan is out), as Plan B-loop in the master session; until serving lands, eval runs in-process through the adapter.
3. **FR3, Aravind** ([part-2 handoff](handoffs/2026-10-23-aravind-part2-fr3-training.md)):
   - SmolVLA on a cloud GPU from `cathlab_fr3_newton_v1@v1.0.1` (~20:30); X-VLA-FR3 comes from the RTX 6000 Ada (~21:30);
   - eval of both + the expert in Newton + MuJoCo → FR3 table + sim2sim gap.
4. **Version v1 → promote the best Panda model → rollout video → failure report → `next_run.json`.** Rakesh, after 1–2.
5. **Panda v2:** merged robosuite + ManiSkill (753 episodes) plus the loop-targeted cells; SmolVLA first, then evaluate and promote if better. RTX 6000 Ada (T6 of the ManiSkill handoff).
6. **Genesis:** Navneet opens a PR from `feat/navneet-genesis-engine`; master session reviews and merges (like PR #10).
7. **CI + Docker:** proposed for Mitesh (free).
8. **Optional:** Newton `v1.1` (+768 episodes).
