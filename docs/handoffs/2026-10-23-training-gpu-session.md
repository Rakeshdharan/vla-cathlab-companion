# Session Handover: VLA fine-tuning on the RTX 6000 Blackwell PC (`pick_robosuite@v1`)

*From the master session, 2026-10-23 ~08:20 (≈H17). Owner: Rakesh. On the GPU PC, paste this file into the agent as the first message, or say "follow `docs/handoffs/2026-10-23-training-gpu-session.md`".*

## Status (2026-10-23 17:30): model set decided

> **Decision (H26.5): fine-tune only ACT, SmolVLA and X-VLA; the scripted expert is the baseline.** π0.5 and MolmoAct2 are **dropped**. Stop the `chain` tmux session if π0.5 is still training, and free the GPU for evaluation and Panda v2. T2 (π0.5), T4 (MolmoAct2) and the π0.5 ablations below are kept for history only.

| Model | Role | State | Steps × batch | HF model repo |
|---|---|---|---|---|
| Scripted expert | baseline (privileged poses) | ✅ robosuite, ✅ ManiSkill (60/60) | — | `engines/<id>/cli eval --policy expert` |
| ACT | non-language lower bound | ✅ on HF | 50,000 × 64 | [`Rakeshdharan56/act_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/act_panda_pick_v1) |
| SmolVLA | small VLA | ✅ on HF | 8,500 × 32 | [`Rakeshdharan56/smolvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/smolvla_panda_pick_v1) |
| X-VLA | frontier VLA | ✅ on HF | 7,500 × 32 | [`Rakeshdharan56/xvla_panda_pick_v1`](https://huggingface.co/Rakeshdharan56/xvla_panda_pick_v1) |
| ~~π0.5~~ / ~~MolmoAct2~~ | — | dropped | — | — |

All three checkpoints were trained on `pick_robosuite@v1`.

**Next on this GPU PC, in order:**
1. **T6:** push `train/lerobot_adapter.py` to `main` (one adapter for all three models).
2. **T7:** closed-loop eval of ACT / SmolVLA / X-VLA + the expert in **robosuite and ManiSkill** (`suite_v2`, seeds 5000–5003).
3. **Panda v2:** retrain the same three on `pick_robosuite@v1` + `pick_maniskill@v1` (753 episodes), with the same recipes and steps.

The pipeline after training (version, rollout, deploy, failure report, next loop) is tracked in [`docs/STATUS.md`](../STATUS.md#the-loop-after-training).

## Goal of this session

Fine-tune four strong VLAs plus one non-VLA baseline on our own synthetic data, evaluate all of them closed-loop in robosuite, and publish them. In order:

| # | Model | Init checkpoint | Why it's in |
|---|---|---|---|
| 1 | **SmolVLA** (450M) | `lerobot/smolvla_libero` | Small, fast, exact LIBERO I/O match. Validates the whole pipeline first |
| 2 | **π0.5** (~3B) | `lerobot/pi05_libero_finetuned` | Frontier; LIBERO avg 97.5% in LeRobot. Exact I/O match |
| 3 | **X-VLA** (0.9B) | `lerobot/xvla-base`, `action_mode=auto` | Frontier, soft-prompted cross-embodiment model |
| 4 | **MolmoAct2** | `allenai/MolmoAct2-LIBERO` | Best LeRobot LIBERO score (98.3% avg), LIBERO "delta EEF" setup like ours |
| 5 | **ACT** (from scratch) | none | Non-VLA lower bound. Not language-conditioned, so expect wrong-object failures. That's the point |

Then:
- one generic **`PolicyAdapter`** in `train/`;
- **closed-loop robosuite eval**: success per object × held-out tier, with the scripted expert as reference;
- **checkpoints on HF, `ModelCard`s, W&B runs**.

**Out of scope:**
- Isaac / FR3 Duo models: different embodiment, and their data doesn't exist yet;
- GR00T N1.5: needs `flash-attn` and `torch<2.8`, which is fragile on Blackwell. Stretch only (T10);
- the loop;
- `serve/`;
- changes to `contracts/`.

## Read first (in repo)

| Doc | Why |
|---|---|
| [`AGENTS.md`](../../AGENTS.md) | Repo rules: contracts frozen, GPU-free default tests, lazy heavy imports, no AI co-author trailers |
| [`docs/specs/2026-10-22-sim-framework-design.md`](../specs/2026-10-22-sim-framework-design.md) | I/O contract (§2), eval tiers (§7), **H8 amendment**: two embodiments, models train per embodiment |
| [`docs/tasks/robosuite-collection.md`](../tasks/robosuite-collection.md) | How the data was made; how to load it |
| [`sim/robosuite/README.md`](../../sim/robosuite/README.md) | The robosuite env + `RobosuiteBackend` used for eval |
| [`docs/decisions.md`](../decisions.md) | Rows H7.5–H9 explain the data, layouts and conventions |
| [`docs/research/vla_selection.md`](../research/vla_selection.md) | History only. **Its I/O numbers (16-D, 224², 30 Hz, chunk 10) and adapter code are pre-H6.5 FR3 values, and its model list is superseded by this file** |

## Decisions already made (don't re-open)

- **Data:** [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite), revision `v1`.
  - Public, 384 episodes (128 per object), 69,191 frames, 20 fps, LeRobot v3.
  - `meta/stats.json` already has mean/std/min/max **and quantiles** (q01–q99), so QUANTILES-normalised models need no conversion.
  - Held-out grid cells are not in the data.
- **I/O (LIBERO-compatible):**

  | Key | Shape | Meaning |
  |---|---|---|
  | `action` | float32[7] | `[dx,dy,dz]/0.05 m`, `[drx,dry,drz]/0.5 rad` (axis-angle, base frame, **relative/delta**), gripper −1 open / +1 close; already in [−1, 1] |
  | `observation.state` | float32[8] | EEF position (base frame, table top z = 0), EEF axis-angle, 2 finger joints |
  | `observation.images.agentview` | 256×256 RGB video | Fixed third-person camera |
  | `observation.images.eye_in_hand` | 256×256 RGB video | Wrist camera |
  | `task` | str | "Hand me the {object}" / "Please pass the {object}" / "Can you give me the {object}" |

- **Model ids:** `smolvla_panda`, `pi05_panda`, `xvla_panda`, `molmoact2_panda`, `act_panda`, all version `v1`.
  - HF model repo: `Rakeshdharan56/<model_id>_pick_v1`.
  - W&B project `cathlab-companion`, run name `<model>-pick_robosuite_v1-<owner>` (AGENTS.md convention).
- **Eval:**
  - tiers = `dataengine.variation.TIERS` (`in_distribution`, `held_out_layout`, `held_out_obj_color`, `held_out_table_color`, `all_held_out`);
  - 3 objects × 5 tiers × 4 seeds (1000–1003) = **60 episodes per model**, max 300 steps;
  - scripted-expert reference row.
- **Time budget:** it's ≈H17; the v0/v1 eval matrix is due H24 and freeze is H32.
  - Each big model gets **≤ 3 h of training wall time**. Set `--steps` from the steps/s measured in its smoke run.
  - **Evaluate each model while the next one trains.** With 96 GB, eval inference fits next to training.

## The GPU: RTX 6000 Blackwell (96 GB, compute capability 12.0 / sm_120)

- **PyTorch must be built for CUDA ≥ 12.8.** Older wheels fail with "no kernel image is available" or `sm_120 is not compatible`.
  - Install torch from the CUDA 12.8 (`cu128`) or newer index.
  - LeRobot 0.5.2 accepts `torch>=2.7,<2.12`.
- Check before anything else:

  ```bash
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'sm_120' in torch.cuda.get_arch_list(), round(torch.cuda.mem_get_info()[1]/2**30), 'GiB')"
  ```

  Expected: CUDA ≥ 12.8, capability `(12, 0)`, `True`, about 95 GiB.
- **96 GB is enough for full fine-tuning of every model here** (MolmoAct2's documented full fine-tune peak is 48–60 GiB at batch 8–32 with gradient checkpointing). Prefer full fine-tunes over LoRA / expert-only, and fall back only if a smoke run OOMs.
- **bf16 everywhere.** If `torch.compile` (`--policy.compile_model=true`) fails or stalls on sm_120, turn it off and log it. It's an optimisation, not a requirement.

## Verified facts (checked 2026-10-23 08:20 on the CPU box)

| Checkpoint | Inputs it expects | Action out | Norm | Chunk | Our mapping |
|---|---|---|---|---|---|
| `lerobot/smolvla_libero` | state [8], `observation.images.camera1`, `camera2` [3,256,256] | [7] | MEAN_STD | 50 | rename `agentview→camera1`, `eye_in_hand→camera2` |
| `lerobot/pi05_libero_finetuned` | state [8], `observation.images.image`, `image2` [3,256,256] + an internal `empty_camera_0` | [7] | MEAN_STD | 50 | rename `agentview→image`, `eye_in_hand→image2` |
| `lerobot/xvla-base` (3.5 GB) | saved processor: `image`, `image2` [256²] + `empty_camera_0`, state [8]; tokenizer `facebook/bart-large`; `xvla_add_domain_id` (0), `xvla_image_to_float`, `xvla_imagenet_normalize` | [20] (`ee6d`) | ACTION MEAN_STD | 30 | rename `agentview→image`, `eye_in_hand→image2`; `--policy.action_mode=auto` (reads our 7-D, pads to 20, loss on the real 7, trims at inference) |
| `allenai/MolmoAct2-LIBERO` (via `--policy.checkpoint_path`) | LeRobot builds the processor **from our dataset metadata**; `--policy.image_keys` names our cameras directly | [7] | QUANTILES (from our stats) | 10 | no rename; `image_keys=[agentview, eye_in_hand]`, `control_mode="delta end-effector pose"` |
| ACT | from dataset | [7] | MEAN_STD | set 50 | none |

**Other facts:**
- `lerobot/xvla-libero` is **not** used. It outputs 20-D absolute `ee6d` actions (LIBERO absolute control mode), and its saved preprocessor lacks the image-to-float and ImageNet steps.
- No ACT LIBERO checkpoint exists.
- **LeRobot 0.5.2 = upstream commit `b06ad4088`** (Python ≥ 3.12):
  - `--rename_map` is supported;
  - `--policy.push_to_hub` defaults to **true**: pass `--policy.repo_id=...` or `--policy.push_to_hub=false`;
  - install extras exist: `smolvla`, `pi`, `xvla`, `molmoact2`.
- **One env for train + eval works:**
  - robosuite 1.5.2 + `mujoco==3.3.7` install into the Python 3.12 LeRobot env;
  - **re-pin `numpy==2.2.6` afterwards** (robosuite's dep `mink` pulls numpy 1.26; mink is GR1-only and its numpy<2 complaint is harmless);
  - all 47 `tests/sim` + `tests/dataengine` tests passed in that setup.
- **`mujoco` must be 3.3.7.** 3.16 breaks robosuite 1.5.2 (`AssertionError` in `get_joint_qpos_addr`).
- Rendering: `MUJOCO_GL=egl`. Images are upright (opencv convention), in the dataset and in the eval env alike.
- Expert reference: 100% on all 10 layouts (train + held-out). Mean episode length 216 / 162 / 163 steps for bottle / vial / syringe, max 263.

## Gotchas (read before coding)

1. **Image orientation.** LIBERO checkpoints saw raw OpenGL frames rotated 180° (`LiberoProcessorStep`), i.e. our upright image mirrored left–right. Fine-tuning absorbs this. **Don't flip anything:** training and eval use the images as stored. Only try a mirror experiment if every VLA lands well below ACT.
2. **Episode length.** `eval.runner.run_suite` has no `max_steps` and inherits `run_episode`'s default of 200. Bottle demos average 216 steps. Call `run_episode(..., max_steps=300)` per config yourself; don't use `run_suite` or `eval/cli.py` for robosuite. Both are frozen API.
3. **Normalisation lives in processors (LeRobot 0.5).** The adapter code in `vla_selection.md` calls `policy.select_action` on raw tensors, which skips normalisation.
   - Always load the checkpoint's own pre/post-processors: `make_pre_post_processors(policy.config, pretrained_path=ckpt)` from `lerobot.policies.factory`.
   - Run `preprocessor(batch)` → `policy.predict_action_chunk(batch)` → `postprocessor(actions)`.
   - The batch keys must be the names **the checkpoint was trained with** after the rename.
4. **Chunking.** Chunk sizes are SmolVLA 50, π0.5 50, X-VLA 30, MolmoAct2 10.
   - Start with each checkpoint's default `n_action_steps`.
   - For the 50-step models, also try 10–25 in eval (T9), because long open-loop chunks correct slowly.
5. **X-VLA.**
   - Keep its saved preprocessor end to end, including `domain_id`. Don't change `domain_id`.
   - Assert the postprocessed chunk is `(n, 7)`.
6. **MolmoAct2.**
   - Use `--policy.checkpoint_path` (original HF weights) with `--policy.type=molmoact2`, **not** `--policy.path`.
   - It builds processors and normalisation stats from our dataset.
   - After fine-tuning, the saved checkpoint loads like any LeRobot policy (`--policy.path` / `from_pretrained`).
7. **`RobosuiteBackend` obs** = `{"state": (8,) float32, "agentview": (256,256,3) uint8, "eye_in_hand": (256,256,3) uint8}`.
   - `reset(scene, seed, target_object)` picks `eval_configs(spec, backend.tier, [seed])` for that object; set `backend.tier` per tier.
   - Reuse one backend per process; env creation takes about 2 s.
8. **Default tests stay GPU-free.** In `train/`, import torch/lerobot inside functions or behind `pytest.importorskip("lerobot")`. Never at module top level in code `uv run pytest` reaches.

## Machine setup (do once)

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks                     # strips AI co-author trailers
git config user.name "<you>" && git config user.email "<you>@..."
uv sync && uv run pytest -q                             # GPU-free suite, must be green

conda create -y -n vla python=3.12 && conda activate vla
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs CUDA >= 12.8 builds
pip install "lerobot[smolvla,pi,xvla,molmoact2] @ git+https://github.com/huggingface/lerobot@b06ad40888ff5045d014d1072ff1b1cac83e4010" \
  pydantic pyyaml imageio pytest wandb
pip install "robosuite==1.5.2" "mujoco==3.3.7" && pip install "numpy==2.2.6"
# re-run the torch check from "The GPU" section: the lerobot install must not have replaced the cu128 torch
hf auth login          # write access to Rakeshdharan56
wandb login            # key from .env (never print or commit it)
export MUJOCO_GL=egl
python -m pytest -q -p no:cacheprovider tests/sim tests/dataengine                  # all pass
```

- **LeRobot 0.5.2 is not on PyPI** (PyPI has 0.5.0/0.5.1/0.6.x). The pin above is the exact upstream commit every flag in this file was verified against. Don't switch to 0.6.x mid-session: flags and processors may differ. If you must, re-verify each command with `lerobot-train --help` and log it.
- If pip replaced torch with a CPU or older-CUDA build, reinstall the `cu128` wheel last.

## Tasks (in order; each ends with a check)

All big-model runs follow the same pattern:
1. **Smoke** (`--steps=200 --save_freq=200 --policy.push_to_hub=false`, separate `--output_dir`). Record VRAM peak (`nvidia-smi`), steps/s, and that a checkpoint is saved.
2. **Full run** with `--steps` = whatever fits ≤ 3 h at the measured steps/s (capped at the step count in each task).
3. **While it trains**, evaluate the previous model (T7).

**T0: Smoke**
- Torch/GPU check (see "The GPU").
- Load the dataset from the Hub and print one frame's shapes.
- Run the robosuite tests.
- Add one row to `docs/decisions.md`: GPU, driver, CUDA, torch, lerobot versions.
- ✅ `LeRobotDataset("Rakeshdharan56/pick_robosuite", revision="v1")` shows 384 episodes, 20 fps, `action (7,)`, `state (8,)`, images `(3,256,256)`.

**T1: SmolVLA** (≤ 20k steps)

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_libero \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --rename_map='{"observation.images.agentview": "observation.images.camera1", "observation.images.eye_in_hand": "observation.images.camera2"}' \
  --batch_size=64 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/train/smolvla_panda_v1 --job_name=smolvla-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/smolvla_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

✅ Loss falls steadily in W&B; checkpoint saved. Record wall time and final loss.

**T2: π0.5** (≤ 10k steps; LeRobot's LIBERO recipe used 6k extra steps)

```bash
lerobot-train \
  --policy.path=lerobot/pi05_libero_finetuned \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --rename_map='{"observation.images.agentview": "observation.images.image", "observation.images.eye_in_hand": "observation.images.image2"}' \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true --policy.compile_model=true \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --batch_size=32 --steps=10000 --save_freq=2500 \
  --output_dir=outputs/train/pi05_panda_v1 --job_name=pi05-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/pi05_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- If the smoke run OOMs, first lower `--batch_size` to 16. Then try `--policy.train_expert_only=true`, and log it.
- If `compile_model` fails on sm_120, set it to `false`.
- ✅ Same checks as T1, plus VRAM peak.

**T3: X-VLA** (≤ 20k steps)

```bash
lerobot-train \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --rename_map='{"observation.images.agentview": "observation.images.image", "observation.images.eye_in_hand": "observation.images.image2"}' \
  --policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --batch_size=32 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/train/xvla_panda_v1 --job_name=xvla-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/xvla_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- The VLM learning rate is automatically 1/10 of the base LR (X-VLA docs).
- **Camera-feature mismatch at startup** (the config lists an `image3`, the saved processor an `empty_camera_0`): first add `--policy.empty_cameras=1`; otherwise make the smallest flag fix and log it.
- ✅ Postprocessed action shape is `(*, 7)`; loss falls.

**T4: MolmoAct2** (≤ 10k steps; the LeRobot LIBERO recipe uses 10k)

```bash
accelerate launch --num_processes=1 --mixed_precision=bf16 -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --dataset.video_backend=pyav --dataset.image_transforms.enable=true \
  --policy.type=molmoact2 --policy.checkpoint_path=allenai/MolmoAct2-LIBERO \
  --policy.device=cuda --policy.action_mode=both \
  --policy.chunk_size=10 --policy.n_action_steps=10 \
  --policy.setup_type="single franka robotic arm in libero" \
  --policy.control_mode="delta end-effector pose" \
  --policy.image_keys='["observation.images.agentview","observation.images.eye_in_hand"]' \
  --policy.model_dtype=bfloat16 --policy.num_flow_timesteps=8 \
  --policy.gradient_checkpointing=true --policy.freeze_embedding=true \
  --policy.normalize_gripper=false --policy.enable_knowledge_insulation=false \
  --policy.repo_id=Rakeshdharan56/molmoact2_panda_pick_v1 \
  --batch_size=32 --steps=10000 --num_workers=4 --log_freq=20 --eval_freq=-1 --save_freq=2000 \
  --output_dir=outputs/train/molmoact2_panda_v1 --job_name=molmoact2-pick_robosuite_v1-<owner> \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- `setup_type` and `control_mode` keep the LIBERO prompt the checkpoint was trained with. Our task is the same embodiment and the same delta-EEF control.
- Documented peak memory for a full fine-tune: 48–60 GiB at batch 8–32. It fits; keep gradient checkpointing on.
- ✅ Loss falls; checkpoint saved.

**T5: ACT from scratch** (50k steps). It's small, so run it **alongside** any of T1–T4:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --batch_size=64 --steps=50000 --save_freq=10000 \
  --output_dir=outputs/train/act_panda_v1 --job_name=act-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/act_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

**T6: Adapter** (`train/lerobot_adapter.py`, `train/README.md`). Do this right after T1, since T7 needs it:
- One class `LeRobotPolicyAdapter(PolicyAdapter)` for all five models:
  - `load(ckpt)` takes a local path or HF repo id. It resolves the policy class from the saved config (`PreTrainedConfig.from_pretrained` → `lerobot.policies.factory.get_policy_class(cfg.type)`) and loads the pre/post-processors (gotcha 3);
  - `predict(obs, instruction)` maps our obs keys to the checkpoint's keys, then returns the postprocessed `(n_action_steps, 7)` float32 chunk clipped to [−1, 1];
  - `meta()` returns name, params (M), chunk size, 20 Hz, action_dim 7;
  - the camera rename map is a constructor argument. Defaults per model type: SmolVLA `camera1`/`camera2`; π0.5 and X-VLA `image`/`image2`; MolmoAct2 and ACT identity.
- Tests in `tests/train/`:
  - a GPU-free unit test (fake policy) for key mapping and output shape/clip;
  - a `pytest.importorskip("lerobot")` test that loads a smoke checkpoint and checks `predict` shapes.
- ✅ `uv run pytest -q` stays green and import-free. The lerobot test passes in the `vla` env.

**T7: Closed-loop robosuite eval** (`scripts/robosuite_eval.py`)
- For each tier in `TIERS` and each seed in `--seeds` (default 1000–1003):
  - set `backend.tier`;
  - call `eval.runner.run_episode(policy, backend, scene, task, obj, seed, max_steps=300)` per object.
- Inputs:
  - `scene = SceneSpec(scene_id="cathlab_v1_robosuite", backend="robosuite", asset="none", objects={...}, cameras=["agentview", "eye_in_hand"])`;
  - `task = load_yaml("configs/tasks/cathlab_pick.yaml", TaskSpec)`.
- Write an `EvalReport` JSON (`contracts.reports`) to `reports/robosuite/<model_id>@v1.json`, with episodes tagged `scene_variant=<tier>`.
- Save a few mp4s to `reports/robosuite/raw/` (git-ignored).
- Add `--policy expert`, a thin wrapper around `run_expert_episode`, for the reference row.
- Report: success by object × tier with Wilson CIs (`eval.metrics`), failed-stage histogram, inference latency p50/p95.
- Then build one comparison table, `reports/robosuite/benchmark_v1.md`, over all five models plus the expert.
- ✅ 60 episodes per model, plus the expert row. Tables go in the PR description and in `docs/decisions.md`.

**T8: Publish**
- Push the final checkpoints to `Rakeshdharan56/<model_id>_pick_v1`. Public is fine: our synthetic data. Keep each upstream model's license in the model card.
- Write `ModelCard` YAMLs (`contracts.reports.ModelCard`) to `reports/robosuite/<model_id>@v1.modelcard.yaml`, with:
  - `adapter: train.lerobot_adapter.LeRobotPolicyAdapter`;
  - `ckpt_uri: hf://Rakeshdharan56/<repo>`;
  - `dataset_versions: ["pick_robosuite@v1"]`;
  - the eval report path, W&B run URL and git sha.
- ✅ Every model loads via `LeRobotPolicyAdapter().load("Rakeshdharan56/<repo>")` on a fresh process.

**T9: Ablations** (only if T1–T8 are done; one decisions row each)
- `n_action_steps` ∈ {10, 25, 50} for SmolVLA and π0.5 on the `all_held_out` tier.
- SmolVLA from `lerobot/smolvla_base` vs `smolvla_libero` at equal steps. Does LIBERO pretraining help? This is a pitch claim.
- π0.5 from `lerobot/pi05_base` vs `pi05_libero_finetuned`.

**T10 (stretch): GR00T N1.5** (`nvidia/GR00T-N1.5-3B`)
- Only if everything else is done.
- It needs `flash-attn` (built against the installed cu128 torch) and LeRobot's GR00T docs pin `torch<2.8`. Use a **separate env**, and time-box the install to 45 min.

## Done = report back to the master session

Open a PR from `feat/<owner>-training-panda-v1` (template filled in), then post a short summary:
- GPU / CUDA / torch versions;
- per model: steps, batch, VRAM peak, wall time, final loss;
- the closed-loop success table (5 models + expert) per object × tier, with CIs;
- inference latency per model;
- HF model links and W&B runs;
- anything that needed a change outside `train/`, `tests/train/`, `scripts/robosuite_eval.py`, `reports/robosuite/` and `docs/`.

## Rules

- Stay in `train/`, `tests/train/`, `scripts/robosuite_eval.py`, `reports/robosuite/*.json|*.md|*.yaml` and docs.
- Don't edit `contracts/` or the frozen `eval/runner.py` API.
- Don't edit `sim/robosuite/` or `dataengine/`. If they need a fix, open a separate small PR and tag Rakesh.
- Checkpoints, datasets and videos never go into git. `outputs/`, `data/` and `reports/**/raw/` are ignored.
- Secrets (HF, W&B) come from `.env` or the CLI logins. Never print, log or commit them.
- Commit as yourself, Conventional Commits, no AI `Co-authored-by` trailers.
- Log every non-obvious choice as one row in `docs/decisions.md`: steps, batch, OOM fallbacks, compile on/off, rename or orientation findings.
