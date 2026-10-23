# Project status: data, models, engines

*Single source of truth for what exists on HF and what is next. Last checked against HF and `main`: **2026-10-23 17:15 (≈ H26)**. Update this file when a dataset or checkpoint lands.*

## Datasets (LeRobot v3)

| Dataset | Robot / I/O | Engine (physics) | Episodes | Frames | Status | Notes |
|---|---|---|---|---|---|---|
| [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) `v1` | Panda, `panda_libero` (7-D EEF delta, 8-D state, 2 cams 256², 20 fps) | robosuite 1.5.2 (MuJoCo 3.3.7) | 384 | 69,191 | ✅ public | Full train grid, expert 100 %. Raw shards: `pick_robosuite_raw` |
| [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) `v1` | Panda, `panda_libero` | ManiSkill3 3.0.1 (**PhysX 5**) | 369 / 384 | 45,387 | ✅ public | Paired with robosuite episode for episode; expert bottle 100 / vial 90 / syringe 100 %. Raw: `pick_maniskill_raw`. [PR #10](https://github.com/Rakeshdharan/vla-cathlab-companion/pull/10) |
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) `v1`, **`v1.0.1`** | FR3 Duo, `fr3_duo_16d` (16-D joint targets, 3 cams 224², 30 fps), right arm | Newton 1.6 (MuJoCo-Warp 3.12) | 384 | 159,568 | ✅ public | Expert 128/128 per object. **Use `v1.0.1` for π0.5 / QUANTILES models**: it widens q01/q99 of the idle left-arm columns (otherwise normalised to ±30). Same episodes as `v1`. [Report](https://github.com/aravinds1610/mujoco_cathlab_vla/blob/main/docs/NEWTON_REPORT.md) |
| [`aravindsuresh16/cathlab_handover_mujoco_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_handover_mujoco_v1) `v3.0` | FR3 Duo, 16-D, **left arm**, his original 6 instruments (not our task) | MuJoCo 3.3.7 | 200 | 91,375 | ⚠️ legacy | Pre-task v1. Not our objects/arm; don't co-train with the above |
| FR3 **MuJoCo** episodes of the team task (CPU twin of the Newton set) | FR3 Duo, 16-D | MuJoCo 3.12 (CPU) | generated, sim2sim only | — | ⏸ **not published** | Same plans as Newton; outcomes agree 60/60. `runner generate --engine mujoco` reproduces them |
| Newton "more episodes" (seeds 1, 2) | FR3 Duo, 16-D | Newton | 768 generated | — | ⏸ **not published** | Resampled physics/light + ±1 cm layout; 768/768 success. Publish as `v1.1` if FR3 models need more data |
| `pick_genesis` | Panda | Genesis 1.4.5 | — | — | ⬜ not started | Handoff ready; optional (PhysX already gives cross-physics) |

## Model checkpoints

| Checkpoint | Base | Trained on | Steps × batch | Status | Trained by |
|---|---|---|---|---|---|
| [`Rakeshdharan56/smolvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_pick_v1) | `lerobot/smolvla_libero` | `pick_robosuite@v1` | 8,500 × 32 | ✅ on HF | RTX 6000 |
| [`Rakeshdharan56/act_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/act_panda_pick_v1) | from scratch | `pick_robosuite@v1` | 50,000 × 64 | ✅ on HF | RTX 6000 |
| [`Rakeshdharan56/xvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_pick_v1) | `lerobot/xvla-base` (`action_mode=auto`) | `pick_robosuite@v1` | 7,500 × 32 | ✅ on HF | RTX 6000 |
| `Rakeshdharan56/pi05_panda_pick_v1` | `lerobot/pi05_libero_finetuned` | `pick_robosuite@v1` | ≤ 10k | ⏳ training (chain) | RTX 6000 |
| `Rakeshdharan56/molmoact2_panda_pick_v1` | `allenai/MolmoAct2-LIBERO` | `pick_robosuite@v1` | ≤ 10k | ⬜ queued | RTX 6000 |
| FR3 models on `cathlab_fr3_newton_v1` | SmolVLA / π0.5 / … | `cathlab_fr3_newton_v1@v1.0.1` | — | ⏳ **Aravind, cloud GPU**: he pushes checkpoints to HF | Aravind |

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

## Blocker for every headline number

**No trained checkpoint has been evaluated closed-loop yet.** Engine `eval --policy url:<server>` needs Sharan's policy server (`serve/`), or the in-process `train/lerobot_adapter.py` from the training session. **Neither is on `main`.** Whichever lands first unblocks the eval matrix: 5 Panda checkpoints × {robosuite, ManiSkill} × suite_v2 tiers, plus FR3 checkpoints in Newton.

## Next (owner)

1. **Land a policy adapter on `main`** (training session on the RTX 6000, or Sharan's server), then evaluate SmolVLA in robosuite and in ManiSkill. Rakesh.
2. **FR3 models on cloud GPU** from `cathlab_fr3_newton_v1@v1.0.1`, pushed to HF. Aravind.
3. **FR3 closed-loop eval** of those checkpoints in Newton (and CPU MuJoCo for sim2sim), using his `evaluate.py` / `PolicyAdapter`. Aravind.
4. **Panda v2 = robosuite + ManiSkill (753 episodes):** retrain SmolVLA and test whether cross-physics co-training helps on held-out tiers (spec D5). RTX 6000, after π0.5 / MolmoAct2.
5. **Optional:** publish Newton `v1.1` (+768 episodes) and the MuJoCo CPU twin if FR3 models need more data or a second FR3 engine for eval. Genesis if someone is free.
