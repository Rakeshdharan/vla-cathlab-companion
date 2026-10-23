# Session Handover: Training on the GPU PC (SmolVLA + X-VLA + ACT on `pick_robosuite@v1`)

*From the master session, 2026-10-23 ~08:15 (≈H17). Owner: Rakesh. On the GPU PC, paste this file into the agent as the first message, or say "follow `docs/handoffs/2026-10-23-training-gpu-session.md`".*

## Goal of this session

Train the first policies on our own synthetic data, evaluate them closed-loop in robosuite, and publish them:

1. **SmolVLA**, fine-tuned from the LIBERO checkpoint `lerobot/smolvla_libero`;
2. **X-VLA** (0.9B, the frontier model), fine-tuned from `lerobot/xvla-base` with the `auto` action mode;
3. **ACT**, trained from scratch (the non-VLA lower bound);
4. a **`PolicyAdapter`** per model in `train/`, so the eval harness and the loop can call them;
5. a **closed-loop robosuite evaluation**: success per object × held-out tier, with the scripted expert as reference;
6. checkpoints on HF, a `ModelCard` per model, runs in W&B.

**Out of scope:**
- Isaac / FR3 Duo models: different embodiment, and their data doesn't exist yet;
- π0.5 (stretch, task T8 only if time and VRAM allow);
- the loop (R8);
- `serve/`;
- changes to `contracts/`.

## Read first (in repo)

| Doc | Why |
|---|---|
| [`AGENTS.md`](../../AGENTS.md) | Repo rules: contracts frozen, GPU-free default tests, lazy heavy imports, no AI co-author trailers |
| [`docs/specs/2026-10-22-sim-framework-design.md`](../specs/2026-10-22-sim-framework-design.md) | I/O contract (§2), eval tiers (§7), **H8 amendment**: two embodiments, models train per embodiment |
| [`docs/tasks/robosuite-collection.md`](../tasks/robosuite-collection.md) | How the data was made; how to load it |
| [`sim/robosuite/README.md`](../../sim/robosuite/README.md) | The robosuite env + `RobosuiteBackend` used for eval |
| [`docs/decisions.md`](../decisions.md) | Why things are the way they are (rows H7.5–H9) |
| [`docs/research/vla_selection.md`](../research/vla_selection.md) | Model lock-in (SmolVLA + ACT for v0; it lists X-VLA as out of scope, but **the H6.5 spec decision D10 makes it the frontier candidate and Rakesh picked it at H17**). **Its I/O numbers (16-D, 224², 30 Hz, chunk 10) and the adapter notes are pre-H6.5 FR3 values. Ignore them; use the table below** |

## Decisions already made (don't re-open)

- **Data:** [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite), revision `v1`.
  - Public, 384 episodes (128 per object), 69,191 frames, 20 fps, LeRobot v3.
  - Full training grid: 8 layouts × 4 object palettes × 4 table colours. Held-out cells are not in the data.
- **I/O (LIBERO-compatible; identical in data, checkpoint and eval):**

  | Key | Shape | Meaning |
  |---|---|---|
  | `action` | float32[7] | `[dx,dy,dz]/0.05 m`, `[drx,dry,drz]/0.5 rad` (axis-angle, base frame), gripper −1 open / +1 close; already in [−1, 1] |
  | `observation.state` | float32[8] | EEF position (base frame, table top z = 0), EEF axis-angle, 2 finger joints |
  | `observation.images.agentview` | 256×256 RGB video | Fixed third-person camera |
  | `observation.images.eye_in_hand` | 256×256 RGB video | Wrist camera |
  | `task` | str | "Hand me the {object}" / "Please pass the {object}" / "Can you give me the {object}" |

- **Models:** SmolVLA (from `lerobot/smolvla_libero`), **X-VLA (from `lerobot/xvla-base`, frontier model)** and ACT (from scratch) now. π0.5 is a stretch goal.
- **Model naming:**
  - model id = `smolvla_panda` / `xvla_panda` / `act_panda`, version `v1` (trained on `pick_robosuite@v1`);
  - W&B project `cathlab-companion`, run name `<model>-pick_robosuite_v1-<owner>` (AGENTS.md convention), e.g. `smolvla-pick_robosuite_v1-rakesh`;
  - HF model repos: `Rakeshdharan56/smolvla_panda_pick_v1`, `Rakeshdharan56/xvla_panda_pick_v1`, `Rakeshdharan56/act_panda_pick_v1`.
- **Eval tiers** (`dataengine.variation.TIERS`): `in_distribution`, `held_out_layout`, `held_out_obj_color`, `held_out_table_color`, `all_held_out`. 3 objects × 5 tiers × 4 seeds = **60 episodes per model**, max 300 steps each. Use seeds that don't overlap training seeds, e.g. 1000–1003.

## Verified facts (checked 2026-10-23 08:10 on the CPU box)

| Fact | Value |
|---|---|
| `lerobot/smolvla_libero` config | `observation.state` [8], `observation.images.camera1` [3,256,256], `observation.images.camera2` [3,256,256], `action` [7], chunk 50, n_action_steps 50, STATE/ACTION MEAN_STD. **Same I/O as ours; only the camera names differ** → `--rename_map` |
| `lerobot/pi05_libero_finetuned` config | cameras `observation.images.image` / `image2` (+ an empty camera), state [8], action [7] |
| `lerobot/xvla-base` (0.9B, 3.5 GB) | Saved preprocessor: tokenizer (`facebook/bart-large`), `xvla_add_domain_id` (domain 0), `xvla_image_to_float`, `xvla_imagenet_normalize`, ACTION MEAN_STD. Its saved features are `observation.images.image` + `image2` (256²), `observation.images.empty_camera_0`, state [8], action [20]: **two real cameras + one empty, same as ours**. Native action mode `ee6d` (20-D) → use `--policy.action_mode=auto` (reads our 7-D, pads to 20, MSE only on the real 7 dims, trims at inference). Chunk 30 |
| `lerobot/xvla-libero` | 20-D `ee6d` actions in LIBERO **absolute** control mode (not our relative deltas), IDENTITY normalisation, and its saved preprocessor **lacks** the image-to-float and ImageNet-normalise steps the X-VLA docs call required. **Not the main init**; T7 ablation only |
| ACT LIBERO checkpoint | None exists (`lerobot/act_libero` missing), so ACT trains from scratch |
| LeRobot | 0.5.2 (from `~/lerobot`, editable). Python ≥ 3.12. `--rename_map` is supported. `--policy.push_to_hub` defaults to **true**: either pass `--policy.repo_id=...` or `--policy.push_to_hub=false` |
| **One env for train + eval** | robosuite 1.5.2 + `mujoco==3.3.7` install into the Python 3.12 LeRobot env. **Re-pin `numpy==2.2.6` afterwards** (robosuite's dep `mink` pulls numpy 1.26; mink is only used by GR1 and its numpy<2 complaint is harmless). All 47 `tests/sim` + `tests/dataengine` tests passed in that env |
| `mujoco` | **Must be 3.3.7.** 3.16 breaks robosuite 1.5.2 (`AssertionError` in `get_joint_qpos_addr`) |
| Rendering | `MUJOCO_GL=egl` (headless). Images come out upright (opencv convention), same as in the dataset |
| Expert reference | Scripted expert: 100% on all 10 layouts (train and held-out), mean 216 / 162 / 163 steps for bottle / vial / syringe, max 263 |
| SmolVLA cost | LeRobot docs: 20k steps ≈ 4 h on one A100 at batch 64 |
| X-VLA fine-tune recipe (LeRobot docs) | `--policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false --policy.train_policy_transformer=true --policy.train_soft_prompts=true`, 20k steps. VLM LR = 1/10 of base, applied automatically |

## Gotchas (read before coding)

1. **Image orientation.** LIBERO checkpoints saw raw OpenGL frames rotated 180° (`LiberoProcessorStep`), which is our upright image mirrored left–right. Fine-tuning absorbs this. **Don't flip anything:** training and eval both use the dataset/env images as they are. Only try a mirror experiment if SmolVLA is clearly worse than ACT.
2. **Episode length.** `eval.runner.run_suite` has no `max_steps` and inherits `run_episode`'s default of 200. Bottle demos average 216 steps. Call `run_episode(..., max_steps=300)` per config yourself; don't use `run_suite` or `eval/cli.py` for robosuite. Both are frozen API, so don't edit them.
3. **Normalisation lives in processors (LeRobot 0.5).** The adapter code in `vla_selection.md` calls `policy.select_action` on raw tensors, which skips the normaliser.
   - Load the checkpoint's pre/post-processors: `make_pre_post_processors(policy.config, pretrained_path=ckpt)` from `lerobot.policies.factory`.
   - Run `preprocessor(batch)` → `policy.predict_action_chunk(batch)` → `postprocessor(actions)`.
   - The batch needs the **checkpoint's** key names (`camera1`/`camera2` for SmolVLA), so apply the same rename the training used.
4. **Chunking.** SmolVLA predicts 50 actions, X-VLA 30. Executing all 50 open-loop is slow to correct. Start with `n_action_steps` 50 (default), then compare 10–25 in eval and log the choice.
5. **`RobosuiteBackend` obs** = `{"state": (8,) float32, "agentview": (256,256,3) uint8, "eye_in_hand": (256,256,3) uint8}`.
   - `reset(scene, seed, target_object)` picks `eval_configs(spec, backend.tier, [seed])` for that object; set `backend.tier` per tier.
   - Reuse one backend per process; env creation takes ~2 s.
6. **X-VLA specifics.**
   - Keep the checkpoint's saved preprocessor end to end (tokenizer, domain id, image-to-float, ImageNet normalise). The adapter must call it, not re-implement it.
   - `domain_id` lives in that preprocessor (`xvla-base` default 0) and is saved with the fine-tuned checkpoint, so train and eval stay consistent automatically. Don't change it.
   - The X-VLA docs list LIBERO as domain 3, but the base model also learns domain 0 for new robots. A domain-3 run is a T7 ablation only.
   - `auto` mode outputs our real 7 dims after postprocessing. Assert `(n, 7)` in the adapter.
7. **Default tests stay GPU-free.** In `train/`, import torch/lerobot inside functions or behind `pytest.importorskip("lerobot")`. Never at module top level in code `uv run pytest` reaches.

## Machine setup (do once)

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks                     # strips AI co-author trailers
git config user.name "<you>" && git config user.email "<you>@..."
uv sync && uv run pytest -q                             # GPU-free suite, must be green (109 passed)

# Training + eval env (Python 3.12)
conda create -y -n lerobot python=3.12 && conda activate lerobot
pip install "lerobot[smolvla,xvla]>=0.5.2,<0.6" pydantic pyyaml imageio pytest wandb
pip install "robosuite==1.5.2" "mujoco==3.3.7" && pip install "numpy==2.2.6"
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info()[1] / 2**30, 'GiB')"

hf auth login          # write access to Rakeshdharan56
wandb login            # key from .env (never print or commit it)
export MUJOCO_GL=egl
MUJOCO_GL=egl python -m pytest -q -p no:cacheprovider tests/sim tests/dataengine   # all pass
```

The `smolvla` and `xvla` extras both exist in 0.5.2 (each pulls `transformers>=5.4,<5.6`). If `lerobot[smolvla,xvla]` resolves to a different minor version, pin `lerobot==0.5.2` instead. The CPU box uses an editable checkout of `huggingface/lerobot` at commit `b06ad4088`.

## Tasks (in order; each ends with a check)

**T0: Smoke**
- Load the dataset from the Hub and print one frame's shapes.
- Run the robosuite tests above.
- Record the GPU model, VRAM, torch/CUDA versions and lerobot version as a row in `docs/decisions.md`.
- ✅ `LeRobotDataset("Rakeshdharan56/pick_robosuite", revision="v1")` shows 384 episodes, 20 fps, `action (7,)`, `state (8,)`, images `(3,256,256)`.

**T1: SmolVLA fine-tune.** Smoke first (200 steps), then the full run:

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_libero \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --rename_map='{"observation.images.agentview": "observation.images.camera1", "observation.images.eye_in_hand": "observation.images.camera2"}' \
  --batch_size=32 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/train/smolvla_panda_v1 --job_name=smolvla-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/smolvla_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- Batch size: start at 32 and raise it to 64 if VRAM allows. Keep `steps × batch_size` around 1.3M samples (≈ 19 epochs).
- If loss plateaus early, stop at the 10k checkpoint and evaluate it.
- If `--rename_map` errors with a "feature mismatch", run `lerobot-train --help`, check the flag shape, and log the fix.
- ✅ The 200-step smoke run finishes and saves a checkpoint. The full run's loss falls steadily in W&B. Record wall time and final loss.

**T2: X-VLA fine-tune** (frontier). Run it after T1, or in parallel only if VRAM allows. Smoke first (200 steps):

```bash
lerobot-train \
  --policy.path=lerobot/xvla-base \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --rename_map='{"observation.images.agentview": "observation.images.image", "observation.images.eye_in_hand": "observation.images.image2"}' \
  --policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --batch_size=16 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/train/xvla_panda_v1 --job_name=xvla-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/xvla_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- 0.9B parameters, fully trainable: start at batch 8–16 in bf16 and measure VRAM in the smoke run.
- If it doesn't fit, freeze the vision encoder first (`--policy.freeze_vision_encoder=true`), then lower the batch size. Log what you used.
- **Camera-feature mismatch at startup** (the config lists an `image3`, the saved processor an `empty_camera_0`):
  - first add `--policy.empty_cameras=1`;
  - if that still fails, read the error, fix it with the smallest flag change, and log it in `docs/decisions.md`.
- ✅ The smoke run saves a checkpoint, and its postprocessed action shape is `(*, 7)`. The full run's loss falls in W&B. Record VRAM, steps/s and wall time.

**T3: ACT from scratch.** It can run alongside T1/T2 if VRAM allows:

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=Rakeshdharan56/pick_robosuite --dataset.revision=v1 \
  --batch_size=64 --steps=50000 --save_freq=10000 \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --output_dir=outputs/train/act_panda_v1 --job_name=act-pick_robosuite_v1-<owner> \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/act_panda_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

- ACT is not language-conditioned. With all three objects present, it can't know which one is wanted, so expect roughly 1/3 "wrong object" outcomes.
- **That's the point of the baseline. Report it; don't "fix" it.**
- ✅ Training finishes; loss curve in W&B.

**T4: Adapters** (`train/lerobot_adapter.py`, plus `train/README.md`)
- `LeRobotPolicyAdapter(PolicyAdapter)`:
  - `load(ckpt)` takes a local path or HF repo id;
  - builds the policy and its pre/post-processors (gotcha 3);
  - `predict(obs, instruction)` maps our obs keys to the checkpoint's keys, then returns the postprocessed `(n_action_steps, 7)` chunk as float32 in [−1, 1];
  - `meta()` returns name, params (M), chunk size, 20 Hz, action_dim 7;
  - the camera rename map is a constructor argument (SmolVLA: `camera1`/`camera2`; X-VLA: `image`/`image2`; ACT: identity);
  - one class for all three: they share LeRobot's `from_pretrained` + `make_pre_post_processors` + `predict_action_chunk` path.
- Tests in `tests/train/`:
  - a GPU-free unit test for the key mapping and output shape/clip, using a fake policy object;
  - a `pytest.importorskip("lerobot")` test that loads a tiny checkpoint (a T1/T2 smoke checkpoint, or `lerobot/smolvla_libero` on CPU) and checks `predict` shapes.
- ✅ `uv run pytest -q` still green and import-free. The lerobot test passes in the `lerobot` env.

**T5: Closed-loop robosuite eval** (`scripts/robosuite_eval.py`)
- For each tier in `TIERS` and each seed in `--seeds` (default 1000–1003):
  - set `backend.tier`;
  - call `eval.runner.run_episode(policy, backend, scene, task, obj, seed, max_steps=300)` per object.
- Inputs:
  - `scene = SceneSpec(scene_id="cathlab_v1_robosuite", backend="robosuite", asset="none", objects={...}, cameras=["agentview", "eye_in_hand"])`;
  - `task = load_yaml("configs/tasks/cathlab_pick.yaml", TaskSpec)`.
- Write an `EvalReport` JSON (`contracts.reports`) to `reports/robosuite/<model>@v1.json`, with episodes tagged `scene_variant=<tier>`.
- Save a few mp4s under `reports/robosuite/raw/`; they are git-ignored.
- Add `--policy expert`: a thin adapter around `run_expert_episode`, or simply the expert check script's per-tier numbers. It gives the reference row.
- Report: success by object × tier with Wilson CIs (`eval.metrics`), failed-stage histogram, and inference latency p50/p95.
- ✅ The report covers 60 episodes per model (SmolVLA, X-VLA, ACT), plus the expert row.
- ✅ Success tables go in the PR description and in `docs/decisions.md`.

**T6: Publish**
- Push the final checkpoints to the HF model repos above. Public is fine: our own synthetic data and Apache-2.0 base weights.
- Write `ModelCard` YAMLs (`contracts.reports.ModelCard`) to `reports/robosuite/<model>@v1.modelcard.yaml`, with:
  - `adapter: train.lerobot_adapter.LeRobotPolicyAdapter`;
  - `ckpt_uri: hf://Rakeshdharan56/<repo>`;
  - `dataset_versions: ["pick_robosuite@v1"]`;
  - the eval report path, W&B run URL and git sha.
- ✅ `LeRobotPolicyAdapter().load("Rakeshdharan56/smolvla_panda_pick_v1")` works on a fresh machine.

**T7: Quick ablations** (only if T1–T6 are done)
- SmolVLA `n_action_steps` ∈ {10, 25, 50} on the `all_held_out` tier.
- SmolVLA from `lerobot/smolvla_base` (non-LIBERO init) vs `smolvla_libero` at equal steps. This tests whether LIBERO pretraining helps: a pitch claim.
- X-VLA from `lerobot/xvla-libero` (auto mode; first check that its saved preprocessor has image normalisation, see verified facts) vs `xvla-base` at equal steps.
- X-VLA `domain_id` 3 (LIBERO) vs 0.
- Log one decisions row each.

**T8 (stretch): π0.5**
- Fine-tune from `lerobot/pi05_libero_finetuned` with rename `agentview → image`, `eye_in_hand → image2`.
- Needs roughly 40 GB+ VRAM for full fine-tuning. Check `docs/source/pi05.mdx` in LeRobot for LoRA/PEFT options first.
- Skip if VRAM is under 24 GB.

## Done = report back to the master session

Open a PR from `feat/<owner>-training-panda-v1` (template filled in), then post a short summary:
- GPU, wall time and steps per model;
- final train loss;
- closed-loop success per object × tier (SmolVLA, X-VLA, ACT, expert), with CIs;
- inference latency;
- HF model links and W&B runs;
- anything that needed a change outside `train/`, `scripts/robosuite_eval.py`, `tests/train/` and `docs/`.

## Rules

- Stay in `train/`, `tests/train/`, `scripts/robosuite_eval.py`, `reports/robosuite/*.json|*.md|*.yaml` and docs.
- Don't edit `contracts/` or the frozen `eval/runner.py` API.
- Don't edit `sim/robosuite/` or `dataengine/`. If they need a fix, open a separate small PR and tag Rakesh.
- Checkpoints, datasets and videos never go into git. `outputs/`, `data/` and `reports/**/raw/` are ignored.
- Secrets (HF, W&B) come from `.env` or the CLI logins. Never print, log or commit them.
- Commit as yourself, Conventional Commits, no AI `Co-authored-by` trailers.
- Log every non-obvious choice as one row in `docs/decisions.md`: batch size, steps, chunking, any rename/orientation finding.
