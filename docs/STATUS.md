# Project status: data, models, engines, codebase

*Single source of truth for what exists on HF and what is next. Last checked against HF and `main`: **2026-10-24 03:50 (≈ H37)**; codebase section at `main` after the H37 stack-finish commits. Update this file when a dataset or checkpoint lands.*

> **Model set (decided H26.5): ACT, SmolVLA, X-VLA, with the scripted expert as the baseline**, for both Panda and FR3. π0.5 and MolmoAct2 are dropped.

## Datasets (LeRobot v3)

| Dataset | Robot / I/O | Engine (physics) | Episodes | Frames | Status | Notes |
|---|---|---|---|---|---|---|
| [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) `v1` | Panda, `panda_libero` (7-D EEF delta, 8-D state, 2 cams 256², 20 fps) | robosuite 1.5.2 (MuJoCo 3.3.7) | 384 | 69,191 | ✅ public | Full train grid, expert 100 %. Raw shards: `pick_robosuite_raw` |
| [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) `v1` | Panda, `panda_libero` | ManiSkill3 3.0.1 (**PhysX 5**) | 369 / 384 | 45,387 | ✅ public | Paired with robosuite episode for episode; expert bottle 100 / vial 90 / syringe 100 %. Raw: `pick_maniskill_raw`. [PR #10](https://github.com/Rakeshdharan/vla-cathlab-companion/pull/10) |
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) `v1`, **`v1.0.1`** | FR3 Duo, `fr3_duo_16d` (16-D joint targets, 3 cams 224², 30 fps), right arm | Newton 1.6 (MuJoCo-Warp 3.12) | 384 | 159,568 | ✅ public | Expert 128/128 per object. **Use `v1.0.1` for π0.5 / QUANTILES models**: it widens q01/q99 of the idle left-arm columns (otherwise normalised to ±30). Same episodes as `v1`. [Report](https://github.com/aravinds1610/mujoco_cathlab_vla/blob/main/docs/NEWTON_REPORT.md) |
| Newton "more episodes" (seeds 1, 2) | FR3 Duo, 16-D | Newton | 768 generated | — | ⏸ **not published** | Resampled physics/light + ±1 cm layout; 768/768 success. Publish as `v1.1` if FR3 models need more data |
| [`Rakeshdharan56/pick_genesis`](https://huggingface.co/datasets/Rakeshdharan56/pick_genesis) `v1` | Panda, `panda_libero` | Genesis 1.4.5 (own rigid solver) | 383 / 384 | 66,447 | ✅ public | Paired with robosuite episode for episode, **plus Genesis-only camera jitter + light direction** (per-episode `genesis_axes` in `episodes_extra.jsonl`); expert bottle 127/128, vial 128/128, syringe 128/128. Raw shards local only |
| `Rakeshdharan56/pick_panda_v2` (local: `data/pick_panda_v2`) | Panda, `panda_libero` | robosuite + ManiSkill + Genesis (merged) | 1,136 | 181,025 | 💾 **local only** (not on the Hub) | `lerobot-edit-dataset` merge of the three v1 sets: episodes 0–383 robosuite, 384–752 ManiSkill, 753–1135 Genesis. Cross-sim co-training data for `xvla_panda_pick_v2` |

## Model checkpoints

| Model | Panda, robosuite data (`pick_robosuite@v1`) | Panda, ManiSkill data (`pick_maniskill@v1`) | FR3 Duo, Newton data (`cathlab_fr3_newton_v1@v1.0.1`) |
|---|---|---|---|
| Scripted expert (baseline) | ✅ robosuite | ✅ ManiSkill 60/60 on suite_v2 | ✅ Newton + MuJoCo 100 % on every tier |
| ACT (from scratch) | ✅ [`act_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/act_panda_pick_v1), 50,000 × 64 (baseline) | dropped | dropped |
| SmolVLA | ✅ [`smolvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_pick_v1), 8,500 × 32, from `smolvla_libero` | ✅ [`smolvla_panda_maniskill_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_maniskill_v1), 8,500 × 32 (v1 recipe), 1 h 54 m, loss 0.011 | ✅ [`aravindsuresh16/smolvla_cathlab_fr3_newton_v1`](https://huggingface.co/aravindsuresh16/smolvla_cathlab_fr3_newton_v1), from `smolvla_base`, **16,000 × 10** (cloud GPU, pushed 21:05; dataset revision not pinned): **Aravind evaluates** |
| X-VLA | ✅ [`xvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_pick_v1), 7,500 × 32, from `xvla-base` (`auto`) | ✅ [`xvla_panda_maniskill_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_maniskill_v1), **2,500 × 32** (time-boxed, v1: 7,500), 36 m, loss 0.028 | ✅ [`xvla_fr3_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_fr3_pick_v1), from `xvla-base` (`auto`, 16-D, 3 cams), **3,500 × 32**, 1 h 08 m, loss 0.001: **Aravind evaluates** |
| ~~π0.5~~, ~~MolmoAct2~~ | dropped | — | dropped |
| **Panda v2** (cross-sim co-training) | SmolVLA v2 ❌ dropped (H28.5). **X-VLA v2** 🔄 `Rakeshdharan56/xvla_panda_pick_v2` on `pick_panda_v2` (robosuite + ManiSkill + Genesis, 1,136 eps), 7,500 × 32, ETA ~04:19 | | — |

**ACT is dropped for the new datasets** (it stays as the robosuite-trained baseline). FR3 checkpoints are evaluated by Aravind in Newton + MuJoCo. The ManiSkill runs use **exactly** the v1 recipes (only the dataset and repo ids differ, proven by a `train_config.json` diff): [`handoffs/2026-10-23-maniskill-training-session.md`](handoffs/2026-10-23-maniskill-training-session.md). Smokes and schedule (RTX 6000 Ada, 48 GB; runs are sequential because no two fit together):

| Run | VRAM peak | s/step (batch 32) | Steps | Start → ETA |
|---|---|---|---|---|
| SmolVLA-ManiSkill | 28.4 GB | 0.78 (AMP) | 8,500 (= v1) | 18:57 → ✅ 20:52 |
| X-VLA-FR3 | 44.1 GB | 1.16 | 3,500 (cut, floor was 7,500) | 20:53 → ✅ 22:03 |
| X-VLA-ManiSkill | 34.0 GB | 0.85 | 2,500 (cut, v1 7,500) | 22:03 → ✅ 22:40 |
| **X-VLA-Genesis** → `Rakeshdharan56/xvla_panda_genesis_v1` | ~34 GB | 1.03 | 7,500 (= v1 recipe) on `pick_genesis@v1` (383 eps, 66k frames) | 23:02 → ✅ 00:53 (1 h 51 m, loss 0.013, 3.6 epochs) |
| **X-VLA v2 (3 engines)** → `Rakeshdharan56/xvla_panda_pick_v2` | 34.0 GB | 0.86 (1.17 step/s) | 7,500 (= v1 recipe) on local `pick_panda_v2` (1,136 eps, 181k frames, ~1.3 epochs) | 02:31 → ~04:20; at 03:24 step 3,687, loss 0.027, ckpt 2,500 saved |

`train_config.json` diff vs v1 (T1): only dataset, job name, output dir, repo id + smoke flags. The handoff's SmolVLA command lacked `--policy.use_amp=true` (v1 had it); the full run sets it. W&B offline as in v1. Evals of finished checkpoints (2–6 GB) run beside training; logs in `outputs/logs/status.txt`.

**Cross-physics eval matrix (the headline, suite_v2, 60 episodes per cell):**

All 21 cells done (robosuite/ManiSkill cells partly on the RTX 2000 Ada, [handoff](handoffs/2026-10-23-rtx2000-cross-eval-session.md); Genesis cells on the RTX 6000 Ada, H32–H35). Next row: `xvla_panda_pick_v2` in all three engines once it lands. Raw reports: `reports/robosuite/eval_reports/`; table: [`reports/robosuite/cross_physics_v1.md`](../reports/robosuite/cross_physics_v1.md).

| | eval robosuite (MuJoCo) | eval ManiSkill (PhysX) | eval Genesis |
|---|---|---|---|
| expert | ✅ 60/60 | ✅ 60/60 | ✅ 60/60 |
| ACT / SmolVLA / X-VLA trained on robosuite | ACT 7 % (3–16) · **SmolVLA 38 % (27–51)** · X-VLA 8 % (4–18): in-domain | ACT 2 % (0–9) · SmolVLA 0 % (0–6) · X-VLA 0 % (0–6) | ACT 0 % (0–6) · SmolVLA 0 % (0–6) · X-VLA 2 % (0–9) |
| SmolVLA / X-VLA trained on ManiSkill | SmolVLA 0 % (0–6) · X-VLA 0 % (0–6) | **SmolVLA 43 % (32–56)** · X-VLA 8 % (4–18): in-domain (RTX 6000 Ada; the RTX 2000 is WSL2 without Vulkan) | SmolVLA 0 % (0–6) · X-VLA 0 % (0–6) |
| X-VLA trained on Genesis (`xvla_panda_genesis_v1`, Genesis camera/light axes in the data) | **18 % (11–30)** | 0 % (0–6) | **27 % (17–39)**: in-domain |

**Reading (22:50, table complete, 12/12 cells):** the cross-physics gap is **symmetric and total**. Each policy works only in the engine it was trained in: SmolVLA 38 % in robosuite / 0 % in ManiSkill when trained on robosuite, and 43 % in ManiSkill / 0 % in robosuite when trained on ManiSkill. Failures in the other engine are at *reach* (55–60/60): the policy doesn't find the object in the other renderer's images. The expert is 100 % in both, so the action/state plumbing is correct. The gap is visual (pedestal, floor, lighting) + dynamics, not an interface bug. X-VLA is weak in-domain in both engines (8 %; the ManiSkill one had only 2,500 steps). **Implication for the loop:** co-train on both engines' data (and/or visual domain randomisation) before claiming sim2sim transfer. SmolVLA-ManiSkill-in-robosuite latency (917 ms p50) was measured on the RTX 2000 laptop with 5 parallel evals. **Update (H35, 21 cells):** the Genesis column is 0–2 % for every off-engine policy, but the Genesis-trained X-VLA (the only dataset with camera/light variation) reaches **18 % in robosuite**, the first transfer above zero (27 % in Genesis, 0 % in ManiSkill); cause not isolated yet (decisions H35).

Training details: [`handoffs/2026-10-23-training-gpu-session.md`](handoffs/2026-10-23-training-gpu-session.md).

**Conventions for new checkpoints** (so eval and the registry find them):
- repo `<user>/<model>_<embodiment>_pick_v<N>`, e.g. `aravindsuresh16/smolvla_fr3_pick_v1`;
- `train_config.json` kept: it records the dataset `repo_id@revision`;
- the model card names the dataset version.

## Engines

| Engine | Robot | Physics | Plugin on `main` | Data | Eval |
|---|---|---|---|---|---|
| robosuite | Panda | MuJoCo 3.3.7 | ✅ `engines/robosuite` | ✅ | ✅ expert + 5 checkpoints (`--policy lerobot:<ckpt>`) |
| ManiSkill3 | Panda | PhysX 5 | ✅ `engines/maniskill` | ✅ | ✅ expert 60/60 + 5 checkpoints |
| Newton | FR3 Duo | MuJoCo-Warp | ⬜ Aravind's repo only ([`cathlab_sim/newton`](https://github.com/aravinds1610/mujoco_cathlab_vla/tree/main/cathlab_sim/newton)) | ✅ | expert 100 % all tiers, sim2sim vs MuJoCo 60/60 |
| Genesis | Panda | Genesis 1.4.5 | ✅ `engines/genesis` (worker :8715, `make worker-genesis`) | ✅ | ✅ expert 60/60; off-engine checkpoints 0–2 %; X-VLA-Genesis 27 % in-domain |
| ~~Isaac Sim~~ | ~~FR3 Duo~~ | ~~PhysX~~ | removed (H24) | — | — |

## Codebase readiness (`main`, H37)

`uv run pytest -q` → **310 passed, 14 skipped** (GPU- and simulator-free; CI runs it on Python 3.10 and 3.12). Simulator tests run in each engine's env (`make conformance-robosuite`, `make conformance-maniskill`, `make conformance-genesis`).

**Legend:** ✅ works end to end · 🟡 partial · ⬜ missing.

| Area | Code | State | What works | What's missing |
|---|---|---|---|---|
| Contracts | `contracts/` | ✅ | TaskSpec, SceneSpec (backends: dummy, robosuite, maniskill, genesis, mujoco, newton, robosimstudio), DatasetManifest, PolicyAdapter, SimBackend, EvalReport (Wilson CI), ModelCard, GapSpec | — |
| Shared sim core | `dataengine/` | ✅ | Variation spec + paired `EpisodeConfig` grid, 5 eval tiers, waypoint expert, fake sim, npz sink, LeRobot sink, generator registry, targeted `cells:` split; **P6 data-quality rules** (`dataengine/quality.py`): no-op frames dropped (LIBERO rule, 0 in our expert data), success must hold 10 steps, shard count checks before merging, provenance per episode | — |
| Engine SDK | `engines/{registry,cli_base,policies,expert_policy}.py` | ✅ | Manifests + I/O profiles (`panda_libero`, `fr3_duo_16d`), uniform CLI (6 stages, `--run`, `--dry-run`, one-line errors), expert-as-policy, conformance test; policies `expert`, `random`, `lerobot:<ckpt>`, `url:<server>` | — |
| robosuite engine | `engines/robosuite`, `sim/robosuite` | ✅ | preview, check-expert, generate, convert, eval (suite_v2), rollout; trained-checkpoint eval through `lerobot:` / `url:` | — |
| ManiSkill3 engine | `engines/maniskill`, `sim/maniskill` | ✅ | All 6 stages; expert 60/60 on suite_v2; worker :8714 (`make worker-maniskill`) | Env only on the RTX 6000 Ada (no `maniskill` env on this box) |
| Genesis engine | `engines/genesis`, `sim/genesis` | ✅ | All 6 stages; expert 60/60 on suite_v2; 5 checkpoints evaluated (0–2 %); `pick_genesis@v1` (383 eps); worker :8715; `CATHLAB_GENESIS_BACKEND=cpu` for a shared GPU | — |
| Newton / MuJoCo FR3 | Aravind's repo | 🟡 | Data on HF, expert 100 %, sim2sim 60/60 | Not an engine plugin in this repo (optional) |
| Control plane + workers | `orchestrator/` | ✅ | Catalog + selection rules, runs (`var/runs/<id>/run.yaml`), SQLite jobs, worker app (token auth, Host allow-list), **stage pipeline** (preview → check expert → generate → convert → **train** → eval → rollout → **register** → **promote**): readiness with reasons, dispatch to workers (train: `train.recipe` on a `train` worker; register/promote in the control plane), logs, cancel, outputs, no double dispatch; `GET /api/datasets` | — |
| Agent harness | `harness/` | ✅ | 12 templates, worktree jobs with path guard + checks + human review, env allow-list, 12 **stage gates**, deterministic **Agent Brief** (UI, CLI `python -m harness brief`, API), **frozen-file deny-list** (`cathlab_v1.yaml` can't be changed by any job) | — |
| UI | `frontend/` | ✅ | PROJECT (catalog, Build with agent), **PIPELINE** (9 stages, live log, outputs, video), **DATA** (published datasets from `configs/datasets.yaml` + local generate outputs), **MODELS** (register, promote, rollback), **REPORT** (failure report, CREATE NEXT RUN), **ASSIST** (procedure step proposes the object, YES hands it over with the production model), AGENT, LOG, **AGENT BRIEF** drawer | — |
| Eval metrics + judge | `eval/` | 🟡 | metrics, failure classes, VLM judge, `eval.cli` (DummySim/robosimstudio), tests | Benchmark table across models × engines; engine evals feed it once checkpoints can be loaded |
| Awareness | `awareness/` | 🟡 | Procedure model, VLM step tracking, propose-confirm assistant (tests); **ASSIST panel wired** (instruction → rollout, YES hands over with the production model) | VLM step tracking not wired to the UI yet (ASSIST proposes from the procedure model, not live video), voice |
| Training | `train/` | ✅ | `train/lerobot_adapter.py` (`--policy lerobot:<ckpt>`, also the policy server's default adapter), **recipes as code** (`train/recipes/{act,smolvla,xvla}.yaml` + `python -m train.recipe`, checked against the real `xvla_panda_genesis_v1` / `xvla_panda_pick_v2` commands), `train/cross_physics_report.py`, `train/model_card.py`; 6 Panda + 2 FR3 checkpoints on HF (`xvla_panda_pick_v2` training) | — |
| Serving + registry | `serve/` | ✅ | Registry (`serve/registry.py`: versions, promotion rule, rollback), policy server (`serve/policy_server.py`, `make policy-server` on :8720) + stdlib `PolicyClient` (`serve/policy_client.py`), codec (`serve/codec.py`) | — |
| Dataset intelligence | `dataset_intel/` | ⬜ | Contract (S5) | Validators, coverage, readiness score; Rakesh |
| Failure report → next run | `report/` | ✅ | `report/failure_report.py`: `build_report` sections (failing cells, sim2sim gap, stage histogram, recommendations) + `next_run.json` (train-split cells only); CREATE NEXT RUN in the REPORT tab | — |
| Loop controller | `orchestrator/loop.py` | ✅ | Next run via the REPORT tab's CREATE NEXT RUN → `dataengine.variation.select_configs` `cells:` split, consumed by every engine's `generate --split cells:<next_run.json>` | — |
| Ingest | `ingest/` | ⬜ | Provided FR3 data on HF (private) | Validation report (low priority) |
| CI + Docker | `.github/workflows/ci.yml`, `docker/control.Dockerfile` | ✅ | Core pytest on Python 3.10/3.12 for every push to `main` and PR (`uv sync --frozen`); control plane + UI image (built and smoke-tested H37) | `compose.yaml` for workers (optional) |
| Docs + agent setup | `AGENTS.md`, `docs/EXTENDING.md`, `docs/HARNESS.md`, `docs/INSTALL.md`, `.github/{instructions,skills,prompts,agents}` | ✅ | Install, harness guide, extending guide (robosuite + ManiSkill examples), engine README, per-area agent instructions, repo skill | — |

**Critical path to the demo:** ✅ adapter · ✅ evals (21 cells) · ✅ registry + promote (real registry H37, production `cathlab-pick@v2`) · ⏳ production rollout videos (queued, H37) · ✅ failure report → `next_run.json` · ✅ ASSIST panel.

## End-to-end check of the loop (H31, RTX 6000 Ada, real checkpoints)

Run on `main` 1efd9d0 with a scratch registry (`var/e2e/registry`), all steps green:

| Step | Command | Result |
|---|---|---|
| Register | `serve.registry register` v1 = `smolvla_panda_pick_v1`, v2 = `smolvla_panda_maniskill_v1`, each with robosuite + ManiSkill eval reports | ✅ 2 versions |
| Promote | `promote cathlab-pick v1`, then `v2` | ✅ v1 production; v2 **refused** (latency p95 7.3 s, measured on the RTX 2000 laptop; it would also fail on 0 % robosuite success) |
| Serve | `serve.policy_server --registry cathlab-pick@production` | ✅ serves `cathlab-pick@v1`; 401 without token |
| Rollout | `engines.robosuite.cli rollout --policy url:http://127.0.0.1:8720 --object vial --seed 5001` | ✅ success in 123 steps, 174 ms/chunk, video |
| Report | `report.failure_report --eval robosuite=… --eval maniskill=… --model cathlab-pick@v1` | ✅ report.md (failing cells, sim2sim gap) + `next_run.json` (2 train-split cells) |
| Next run | `engines.maniskill.cli generate --split cells:next_run.json --n 4` | ✅ 4/4 kept, only the failing cells, distinct seeds/physics |

Gotchas: on machines with an HTTP proxy, set `NO_PROXY=127.0.0.1,localhost` (otherwise localhost calls get 407 and 4 core tests fail: `tests/serve/test_policy_server.py`, `tests/orchestrator/test_control.py`). The policy server's env needs `fastapi uvicorn` next to lerobot.

## The loop after training

Every stage below already has a contract (spec Part B P9–P12 and Sharan's [handoff](handoffs/2026-10-23-sharan-serve-intel-session.md)). **What's missing is the code on `main`**:

| Stage | What it produces | Owner | Code on `main` | State |
|---|---|---|---|---|
| **Eval** | `EvalReport` per model × engine × tier (suite_v2: 3 objects × 5 tiers × seeds 5000–5003 = 60 episodes); expert as reference | Rakesh (Panda), Aravind (FR3) | ✅ `engines/{robosuite,maniskill,genesis}/cli eval` with `lerobot:` / `url:` policies | ✅ 12 cells (6 models × robosuite/ManiSkill): [`reports/robosuite/cross_physics_v1.md`](../reports/robosuite/cross_physics_v1.md) |
| **Version** | Registry entry `cathlab-pick@vN` = ModelCard (checkpoint URI, dataset versions, eval report, W&B run, git SHA) | Rakesh (was Sharan, S3) | ✅ `serve/registry.py` | ✅ |
| **Promote / rollback** | Production pointer moves only if the candidate beats production beyond the 95 % CI and p95 latency ≤ 500 ms (`configs/serve.yaml`) | Rakesh (was Sharan, S3) | ✅ `serve/registry.py` | ✅ |
| **Deploy** | Policy server serving the production version (HTTP+JSON, token auth) + stdlib `PolicyClient` used by every engine | Rakesh (was Sharan, S1, S2) | ✅ `serve/policy_server.py`, `serve/policy_client.py` | ✅ |
| **Rollout** | One live episode of the production model with video, from the Control Station (ASSIST/PIPELINE) | Rakesh | ✅ `engines/<id>/cli rollout --policy url:<server>` | ✅ e2e on a scratch registry (H31); production rollout videos in 3 engines queued after `xvla_panda_pick_v2` (H37) |
| **Failure report → next loop** | Failing cells (engine × tier × object × layout/colour), stage histogram, wrong-object grasps, sim2sim gap, `next_run.json` with exactly those cells → generate → v2 data → retrain | Rakesh (was Sharan, S6) | ✅ `report/failure_report.py` | ✅ |

## Next (owner)

**Demo finish moves to the RTX 2000 Ada:** [`handoffs/2026-10-24-rtx2000-demo-finish-session.md`](handoffs/2026-10-24-rtx2000-demo-finish-session.md) (review R1–R7, demo script, `make demo-bootstrap`). The RTX 6000 Ada keeps the GPU jobs (rollout videos, targeted generate, `xvla_panda_pick_v2` evals).

Done since H31: policy adapter + server, registry/promote/rollback, failure report → `next_run.json`, the 21-cell cross-physics table, the Genesis engine + data + column, recipes as code, P6 data-quality rules, DATA tab, train/register/promote PIPELINE stages, frozen-file guard, CI + Docker.

1. **Real registry (H37):** `cathlab-pick@v1..v6` registered with robosuite + ManiSkill + Genesis evals in `var/registry` of the RTX 6000 Ada checkout; **production = v2 (`smolvla_panda_pick_v1`, 38 % robosuite)** by the rule (v1 ACT promoted first, then replaced; v6 X-VLA-Genesis refused, CI low 11 % ≤ 38 %). Failure report + `next_run.json` for `cathlab-pick@v2` in `var/reports/cathlab-pick_v2` (2 train cells × 8). **Queued after `xvla_panda_pick_v2` (~04:19):** targeted generate in all three engines and production rollout videos (robosuite, ManiSkill, Genesis × 3 objects) → `var/loop_v2/`. Navneet.
2. **`xvla_panda_pick_v2`** (cross-sim co-training): eval in robosuite + ManiSkill + Genesis, register, promote if it beats v2. Rakesh / Navneet.
3. **FR3, Aravind:** eval `smolvla_cathlab_fr3_newton_v1` and `xvla_fr3_pick_v1` + the expert in Newton + MuJoCo → FR3 table + sim2sim gap.
4. **Ablation for the 18 % transfer:** regenerate Genesis data with `--no-genesis-axes`, same X-VLA recipe, re-evaluate (decisions H35).
5. **Optional:** `dataset_intel/` (validators, coverage, readiness score), `ingest/` validation report, awareness VLM step tracking in ASSIST + voice, Newton `v1.1` (+768 episodes), publish `pick_panda_v2` to the Hub.
