# Project status: data, models, engines

*Single source of truth for what exists on HF and what is next. Last checked against HF and `main`: **2026-10-23 17:30 (≈ H26.5)**. Update this file when a dataset or checkpoint lands.*

> **Model set (decided H26.5): ACT, SmolVLA, X-VLA, with the scripted expert as the baseline**, for both Panda and FR3. π0.5 and MolmoAct2 are dropped.

## Datasets (LeRobot v3)

| Dataset | Robot / I/O | Engine (physics) | Episodes | Frames | Status | Notes |
|---|---|---|---|---|---|---|
| [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) `v1` | Panda, `panda_libero` (7-D EEF delta, 8-D state, 2 cams 256², 20 fps) | robosuite 1.5.2 (MuJoCo 3.3.7) | 384 | 69,191 | ✅ public | Full train grid, expert 100 %. Raw shards: `pick_robosuite_raw` |
| [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) `v1` | Panda, `panda_libero` | ManiSkill3 3.0.1 (**PhysX 5**) | 369 / 384 | 45,387 | ✅ public | Paired with robosuite episode for episode; expert bottle 100 / vial 90 / syringe 100 %. Raw: `pick_maniskill_raw`. [PR #10](https://github.com/Rakeshdharan/vla-cathlab-companion/pull/10) |
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) `v1`, **`v1.0.1`** | FR3 Duo, `fr3_duo_16d` (16-D joint targets, 3 cams 224², 30 fps), right arm | Newton 1.6 (MuJoCo-Warp 3.12) | 384 | 159,568 | ✅ public | Expert 128/128 per object. **Use `v1.0.1` for π0.5 / QUANTILES models**: it widens q01/q99 of the idle left-arm columns (otherwise normalised to ±30). Same episodes as `v1`. [Report](https://github.com/aravinds1610/mujoco_cathlab_vla/blob/main/docs/NEWTON_REPORT.md) |
| [`aravindsuresh16/cathlab_handover_mujoco_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_handover_mujoco_v1) `v3.0` | FR3 Duo, 16-D, **left arm**, his original 6 instruments (not our task) | MuJoCo 3.3.7 | 200 | 91,375 | ⚠️ legacy | Pre-task v1. Not our objects/arm; don't co-train with the above. Aravind adds a "legacy" banner |
| `aravindsuresh16/cathlab_fr3_mujoco_v1` (team task, CPU twin of the Newton set) | FR3 Duo, 16-D | MuJoCo 3.12 (CPU) | 384 (planned) | — | ⏳ **Aravind, by 18:30** ([part-2 handoff](handoffs/2026-10-23-aravind-part2-mujoco-and-training.md)) | Same task/objects/grid/schema as Newton; episode *i* = Newton episode *i*. Replaces the legacy MuJoCo set |
| Newton "more episodes" (seeds 1, 2) | FR3 Duo, 16-D | Newton | 768 generated | — | ⏸ **not published** | Resampled physics/light + ±1 cm layout; 768/768 success. Publish as `v1.1` if FR3 models need more data |
| `pick_genesis` | Panda | Genesis 1.4.5 | — | — | ⬜ not started | Handoff ready; optional (PhysX already gives cross-physics) |

## Model checkpoints

| Model | Panda (`pick_robosuite@v1`, RTX 6000) | FR3 Duo (`cathlab_fr3_newton_v1@v1.0.1`, Aravind, cloud GPU) |
|---|---|---|
| Scripted expert (baseline) | ✅ robosuite; ✅ ManiSkill 60/60 on suite_v2 | ✅ Newton + MuJoCo 100 % on every tier |
| ACT (from scratch) | ✅ [`Rakeshdharan56/act_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/act_panda_pick_v1), 50,000 × 64 | ⏳ `aravindsuresh16/act_fr3_pick_v1` |
| SmolVLA | ✅ [`Rakeshdharan56/smolvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_pick_v1), 8,500 × 32, from `smolvla_libero` | ⏳ `aravindsuresh16/smolvla_fr3_pick_v1`, from `smolvla_base`, ≥ 15k × 64 (first, on HF by 20:30) |
| X-VLA | ✅ [`Rakeshdharan56/xvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_pick_v1), 7,500 × 32, from `xvla-base` (`action_mode=auto`) | ⏳ `aravindsuresh16/xvla_fr3_pick_v1`, from `xvla-base` (`auto`, 16-D) |
| ~~π0.5~~, ~~MolmoAct2~~ | dropped (stop the π0.5 chain on the RTX 6000) | dropped |
| **Panda v2** (all three) | ⬜ after eval: `pick_robosuite@v1` + `pick_maniskill@v1` (753 episodes) → `*_panda_pick_v2` | — |

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
| Genesis | Panda | Genesis | ⬜ | ⬜ | ⬜ |
| ~~Isaac Sim~~ | ~~FR3 Duo~~ | ~~PhysX~~ | removed (H24) | — | — |

## The loop after training

Every stage below already has a contract (spec Part B P9–P12 and Sharan's [handoff](handoffs/2026-10-23-sharan-serve-intel-session.md)). **What's missing is the code on `main`**:

| Stage | What it produces | Owner | Code on `main` | State |
|---|---|---|---|---|
| **Eval** | `EvalReport` per model × engine × tier (suite_v2: 3 objects × 5 tiers × seeds 5000–5003 = 60 episodes); expert as reference | Rakesh (Panda), Aravind (FR3) | ✅ `engines/{robosuite,maniskill}/cli eval` (expert works); **needs a policy adapter** | ⛔ blocked on `train/lerobot_adapter.py` or `serve/policy_client.py` |
| **Version** | Registry entry `cathlab-pick@vN` = ModelCard (checkpoint URI, dataset versions, eval report, W&B run, git SHA) | Sharan (S3) | ⬜ `serve/registry.py` | not started on `main` |
| **Promote / rollback** | Production pointer moves only if the candidate beats production beyond the 95 % CI and p95 latency ≤ 200 ms (`configs/serve.yaml`) | Sharan (S3) | ⬜ | — |
| **Deploy** | Policy server serving the production version (HTTP+JSON, token auth) + stdlib `PolicyClient` used by every engine | Sharan (S1, S2) | ⬜ `serve/policy_server.py`, `serve/policy_client.py` | — |
| **Rollout** | One live episode of the production model with video, from the Control Station (ASSIST/PIPELINE) | Rakesh | ✅ `engines/<id>/cli rollout` (expert) | waits for Deploy |
| **Failure report → next loop** | Failing cells (engine × tier × object × layout/colour), stage histogram, wrong-object grasps, sim2sim gap, `next_run.json` with exactly those cells → generate → v2 data → retrain | Sharan (S6) + Rakesh (loop) | ⬜ `report/failure_report.py` | — |

## Blocker for every headline number

**No trained checkpoint has been evaluated closed-loop yet.** Engine `eval --policy url:<server>` needs Sharan's policy server (`serve/`), or the in-process `train/lerobot_adapter.py` from the training session. **Neither is on `main`.** Whichever lands first unblocks the eval matrix: 5 Panda checkpoints × {robosuite, ManiSkill} × suite_v2 tiers, plus FR3 checkpoints in Newton.

## Next (owner)

1. **Unblock eval:** push `train/lerobot_adapter.py` to `main` (RTX 6000 session), then eval ACT / SmolVLA / X-VLA + the expert in robosuite and ManiSkill. **Rakesh.**
2. **Serving + registry:** policy server/client (S1–S2), registry with promote/rollback (S3), failure report (S6). **Sharan.** Ask for an ETA; until it lands, eval runs in-process through the adapter.
3. **FR3, Aravind** ([part-2 handoff](handoffs/2026-10-23-aravind-part2-mujoco-and-training.md)):
   - `cathlab_fr3_mujoco_v1` by 18:30;
   - SmolVLA → X-VLA → ACT on a cloud GPU from `cathlab_fr3_newton_v1@v1.0.1`;
   - eval in Newton + MuJoCo → FR3 table + sim2sim gap.
4. **Version v1 → promote the best Panda model → rollout video → failure report → `next_run.json`.** Rakesh + Sharan, after 1–2.
5. **Panda v2:** robosuite + ManiSkill (753 episodes) plus the loop-targeted cells, retrain the three models, evaluate, promote if better. RTX 6000.
6. **Optional:** Newton `v1.1` (+768 episodes); Genesis.
