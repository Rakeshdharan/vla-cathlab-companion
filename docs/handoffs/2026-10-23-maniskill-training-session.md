# Session Handover: RTX 6000 Ada: SmolVLA + X-VLA on ManiSkill3 data, X-VLA on FR3 Newton data → cross-physics eval

*From the master session, 2026-10-23 ~17:45 (≈ H26.5). Owner: Rakesh. Machine: the **RTX 6000 Ada (48 GB)** workstation that trained the robosuite v1 models and built the ManiSkill engine. Paste this file into the agent there as the first message, or say "follow `docs/handoffs/2026-10-23-maniskill-training-session.md`".*

## 1. Goal

> **Update 17:55: three runs on this GPU, ACT dropped** for the new datasets. ACT stays only as the robosuite-trained baseline (`act_panda_pick_v1`, already on HF).
>
> | Run | Dataset | Checkpoint |
> |---|---|---|
> | SmolVLA | `Rakeshdharan56/pick_maniskill@v1` (Panda, PhysX) | `Rakeshdharan56/smolvla_panda_maniskill_v1` |
> | X-VLA | `Rakeshdharan56/pick_maniskill@v1` | `Rakeshdharan56/xvla_panda_maniskill_v1` |
> | **X-VLA (FR3 Duo)** | **`aravindsuresh16/cathlab_fr3_newton_v1@v1.0.1`** (Newton, 16-D, 3 cams) | **`Rakeshdharan56/xvla_fr3_pick_v1`** |
>
> FR3 SmolVLA is trained by Aravind (on HF by ~20:30). **Aravind evaluates both FR3 checkpoints** in Newton + MuJoCo (his stack). This machine evaluates the Panda checkpoints.

Train SmolVLA and X-VLA on **ManiSkill3 (PhysX 5) data** with **exactly the recipes of the robosuite (MuJoCo) v1 models**, then evaluate all Panda checkpoints plus the expert in **both engines**. The result is the project's cross-physics table:

| | eval in robosuite (MuJoCo) | eval in ManiSkill (PhysX) |
|---|---|---|
| Scripted expert (baseline) | ✅ | ✅ 60/60 |
| ACT / SmolVLA / X-VLA **trained on robosuite** (`*_panda_pick_v1`, on HF) | in-domain | **cross-physics** |
| SmolVLA / X-VLA **trained on ManiSkill** (`*_panda_maniskill_v1`, this session) | **cross-physics** | in-domain |

**Model set: ACT (robosuite baseline only), SmolVLA, X-VLA + the scripted expert as baseline.** π0.5 and MolmoAct2 are dropped.

## 2. Facts (verified 2026-10-23 17:40 from HF)

| Fact | Value |
|---|---|
| Dataset | [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) **revision `v1`**: 369 episodes (paired with robosuite configs), 45,387 frames, 20 fps, `robot_type panda`. Same keys and shapes as `pick_robosuite`: `observation.state` (8), `observation.images.agentview` + `observation.images.eye_in_hand` (256×256), `action` (7) |
| Video decoding | Use `--dataset.video_backend=pyav`. The default decoder is broken on this machine, and the v1 runs used pyav too |
| The v1 recipes (from each checkpoint's `train_config.json` on HF) | **SmolVLA:** `--policy.path=lerobot/smolvla_libero`, rename `agentview→camera1`, `eye_in_hand→camera2`, 8,500 steps × batch 32, save every 2,500, 24 workers, seed 1000, chunk 50.<br>**ACT:** `--policy.type=act`, chunk 50 / n_action_steps 50, 50,000 × 64, save every 10,000, 16 workers, seed 1000.<br>**X-VLA:** `--policy.path=lerobot/xvla-base`, rename `agentview→image`, `eye_in_hand→image2`, bf16, `action_mode=auto`, `max_action_dim=20`, all encoders trainable + policy transformer + soft prompts, 7,500 × 32, save every 5,000, 16 workers, seed 1000 |
| Fair comparison | **Same steps × batch as v1.** ManiSkill has 45k frames vs robosuite's 69k (shorter episodes), so each model sees ≈ 1.5× more epochs. Note it in the model card; don't "correct" it |
| LeRobot | The existing LeRobot env on this machine (`/media/data1/waifzi2/vla-env`, used for the v1 runs and the ManiSkill conversion). Keep the same LeRobot version as v1, and check that `lerobot-train --help` accepts every flag below |
| Multi-dataset training | **Not supported** by `lerobot-train` (it raises `MultiLeRobotDataset isn't supported`). Co-training (T6) merges datasets first with `lerobot-edit-dataset --operation.type merge` |
| GPU | 48 GB (the v1 runs had more headroom). Driver 570 → torch **cu128** builds |

## 3. Decisions (don't re-open)

| # | Decision |
|---|---|
| M1 | Checkpoints `Rakeshdharan56/{smolvla,xvla}_panda_maniskill_v1` and `Rakeshdharan56/xvla_fr3_pick_v1`. W&B project `cathlab-companion`, runs `<model>-pick_maniskill_v1-rakesh` and `xvla-fr3_newton_v1.0.1-rakesh` |
| M2 | **ManiSkill runs:** the only differences from v1 are `--dataset.repo_id`, `--output_dir`, `--job_name`, `--policy.repo_id`. Prove it by diffing the new `train_config.json` against v1's (T1). **The FR3 X-VLA run** also changes the camera rename (3 cameras), the dataset revision (`v1.0.1`) and possibly the steps (time-boxed, §5) |
| M3 | Eval = `configs/eval/suite_v2.yaml`: 3 objects × 5 tiers × seeds 5000–5003 = 60 episodes per checkpoint per engine, max 400 steps, the shared expert as reference. Output `contracts.EvalReport` JSON with `scene_variant` = tier |
| M4 | Data and checkpoints never go into git. Commit as yourself, with no AI `Co-authored-by` trailers. Rebase on `main` before pushing (`main` moves fast) |

## 4. Tasks (in order; each ends with a ✅)

**T0: Preflight (10 min)**

```bash
cd <repo> && git pull --rebase
source /media/data1/waifzi2/vla-env/bin/activate        # the v1 training env
python -c "import torch, lerobot; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), round(torch.cuda.mem_get_info()[1]/2**30), 'GiB', lerobot.__version__)"
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d = D('Rakeshdharan56/pick_maniskill', revision='v1', video_backend='pyav'); f = d[0]; print(d.num_episodes, d.fps, {k: tuple(f[k].shape) for k in ('action', 'observation.state', 'observation.images.agentview')})"
tmux ls; nvidia-smi        # make sure the old pi0.5 `chain` session is stopped
```

✅ 369 episodes, 20 fps, `(7,)`, `(8,)`, `(3, 256, 256)`; the GPU is free.

**T1: Smoke all three runs (200 steps each, no push)**

Run each command in §5 with `--steps=200 --save_freq=200 --policy.push_to_hub=false --wandb.enable=false` and `--output_dir=outputs/smoke/<run>` (`smolvla_maniskill`, `xvla_maniskill`, `xvla_fr3`). For each, record VRAM peak (`nvidia-smi`) and steps/s, then diff its config against v1:

```bash
python - <<'EOF'
import json
from huggingface_hub import hf_hub_download

def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out

for m in ("smolvla", "xvla"):
    v1 = flat(json.load(open(hf_hub_download(f"Rakeshdharan56/{m}_panda_pick_v1", "train_config.json"))))
    new = flat(json.load(open(f"outputs/smoke/{m}_maniskill/checkpoints/last/pretrained_model/train_config.json")))
    print(m, {k: (v1.get(k), new.get(k)) for k in sorted(v1.keys() | new.keys()) if v1.get(k) != new.get(k)})
EOF
```

✅ All three smokes save a checkpoint. For the two ManiSkill runs, the **only** diffs are dataset repo, output dir, job name, policy repo id, and the smoke's own steps/save/push/wandb flags. For `xvla_fr3`, check that the postprocessed action is `(*, 16)`. Write VRAM and steps/s into `docs/STATUS.md` (training table).

**T2: Full runs**
- **Order:** SmolVLA-ManiSkill + X-VLA-FR3 together if the smoke VRAM allows, otherwise SmolVLA-ManiSkill + X-VLA-ManiSkill first. Then the remaining X-VLA. Two X-VLA runs at batch 32 will likely not fit 48 GB together, so check the smoke VRAM.
- **Time-box:** everything must be on HF by ~22:00 so Aravind can evaluate X-VLA-FR3 and you can finish the Panda eval before freeze.
  - Compute each run's steps from the smoke's steps/s.
  - ManiSkill runs keep the v1 steps (SmolVLA 8,500, X-VLA 7,500) unless they can't finish by 22:00; then cut steps and log it.
  - X-VLA-FR3: see §5.
- If an X-VLA run OOMs at batch 32, run batch 16 with twice the steps (same samples seen) and log it in `docs/decisions.md`.
- Run each in its own `tmux` session (`ms-smolvla`, `ms-xvla`, `fr3-xvla`) with a log in `outputs/logs/<checkpoint>.log`.
- Keep `outputs/logs/status.txt` refreshed every 10 min (run, step / total, loss, steps/s, ETA).

✅ Each run finishes, auto-pushes to its repo (M1) and loads from the Hub in a fresh process. Model card: dataset + tag, base checkpoint, steps × batch, GPU, wall time, final loss. **Update `docs/STATUS.md`.**
- **When `xvla_fr3_pick_v1` is on HF, tell Rakesh**, who tells Aravind (he evaluates it).

**T3: Eval environment (while training runs)**

One Python 3.12 env with LeRobot and both simulators. This combination was verified on 2026-10-23: robosuite 1.5.2 and mani_skill 3.0.1 both install next to LeRobot on 3.12, and all `tests/sim` + `tests/dataengine` tests pass. **Don't touch the training env:**

```bash
python3.12 -m venv --system-site-packages /media/data1/waifzi2/vla-eval     # inherits torch/lerobot from vla-env's interpreter
source /media/data1/waifzi2/vla-eval/bin/activate
pip install "mani_skill==3.0.1" "robosuite==1.5.2" "mujoco==3.3.7" && pip install "numpy==2.2.6"   # robosuite's mink pulls numpy 1.26; re-pin
cd <repo> && MUJOCO_GL=egl python -m pytest -q -p no:cacheprovider tests/sim tests/engines
```

- If `--system-site-packages` doesn't pick up `vla-env`'s packages (a different base interpreter), create the venv **from `vla-env`'s own python** (`/media/data1/waifzi2/vla-env/bin/python -m venv --system-site-packages ...`).
- ✅ The robosuite + ManiSkill engine and conformance tests pass in `vla-eval`.

**T4: Policy adapter + `--policy lerobot:<ckpt>`.** Skip this if `train/lerobot_adapter.py` has already landed on `main` (`git pull` first).
- `train/lerobot_adapter.py`: `LeRobotPolicyAdapter(PolicyAdapter)`, spec in [`2026-10-23-training-gpu-session.md`](2026-10-23-training-gpu-session.md) T6. One class for all three models:
  - resolve the class from the checkpoint config: `PreTrainedConfig.from_pretrained(ckpt)` → `lerobot.policies.factory.get_policy_class(cfg.type)`;
  - load processors with `make_pre_post_processors(cfg, pretrained_path=ckpt)`;
  - feed the **dataset key names** (`observation.state`, `observation.images.agentview`, `observation.images.eye_in_hand`) plus `task` = instruction. The checkpoint's saved preprocessor already applies the training camera rename (`lerobot_train` stores `rename_map` in `rename_observations_processor`), so **don't rename in the adapter**. This is the Panda adapter; FR3 eval happens in Aravind's stack;
  - images HWC uint8 → CHW float in [0, 1];
  - `predict_action_chunk` → postprocess → `(n_action_steps, 7)` float32 clipped to [−1, 1].
- `engines/policies.py`: add `lerobot:<repo-or-path>` to `make_policy`, with a lazy import of the adapter. That is the only change outside `train/`.
- Tests:
  - a GPU-free unit test (fake policy) for key mapping and output shape/clip, in the core suite;
  - one `pytest.importorskip("lerobot")` test that loads `Rakeshdharan56/act_panda_pick_v1` (the smallest model) and checks the `predict` shape.
- Small PR, or rebase and push to `main`. `uv run pytest -q` must stay green and simulator-free.
- ✅ `python -m engines.robosuite.cli rollout --policy lerobot:Rakeshdharan56/act_panda_pick_v1 --out var/smoke/act_rollout` runs one episode and writes a video.

**T5: Cross-physics eval (the headline)**

In `vla-eval`, for each checkpoint × engine:

```bash
for ENG in robosuite maniskill; do
  python -m engines.$ENG.cli eval --suite configs/eval/suite_v2.yaml --policy expert --model-version expert --out var/eval/$ENG/expert
  for CK in act_panda_pick_v1 smolvla_panda_pick_v1 xvla_panda_pick_v1 smolvla_panda_maniskill_v1 xvla_panda_maniskill_v1; do
    python -m engines.$ENG.cli eval --suite configs/eval/suite_v2.yaml --policy lerobot:Rakeshdharan56/$CK \
      --model-version $CK --video failures --out var/eval/$ENG/$CK
  done
done
```

- **Start with the robosuite-trained checkpoints while the ManiSkill ones train**: those three are already on HF.
- Run evals on the same GPU as training only if VRAM allows: inference uses ~2–6 GB per model.
- ✅ 12 `eval_report.json` files (6 per engine: the expert + 5 checkpoints, 60 episodes each).
- ✅ One table, `reports/robosuite/cross_physics_v1.md`: success per model × train source × eval engine, overall and per tier, with Wilson 95 % CIs (`eval.metrics`), plus the failed-stage histogram and latency p50/p95.
- Copy the summary into `docs/STATUS.md` and one row into `docs/decisions.md`.
- Keep failure videos in `var/` (git-ignored). They feed the failure report → next loop.

**T6 (only if T5 is done by ~21:30): Panda v2 co-training**

```bash
lerobot-edit-dataset --new_repo_id Rakeshdharan56/pick_panda_v2 --new_root data/pick_panda_v2 \
  --operation.type merge --operation.repo_ids "['Rakeshdharan56/pick_robosuite', 'Rakeshdharan56/pick_maniskill']"
```

- Check the merged set loads: 753 episodes, same features. Push it with tags `v1` + `v3.0`.
- Train **SmolVLA** first with the v1 recipe on `pick_panda_v2` → `Rakeshdharan56/smolvla_panda_pick_v2`, then evaluate in both engines (T5).
- The claim it tests (spec D5): does co-training on two physics engines beat single-engine training on the *other* engine and on the held-out tiers?

## 5. Training commands

```bash
# SmolVLA on ManiSkill (exact v1 recipe)
lerobot-train --policy.path=lerobot/smolvla_libero \
  --dataset.repo_id=Rakeshdharan56/pick_maniskill --dataset.revision=v1 --dataset.video_backend=pyav \
  --rename_map='{"observation.images.agentview": "observation.images.camera1", "observation.images.eye_in_hand": "observation.images.camera2"}' \
  --batch_size=32 --steps=8500 --save_freq=2500 --num_workers=24 --seed=1000 \
  --output_dir=outputs/train/smolvla_panda_maniskill_v1 --job_name=smolvla-pick_maniskill_v1-rakesh \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/smolvla_panda_maniskill_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion

# X-VLA on ManiSkill (exact v1 recipe)
lerobot-train --policy.path=lerobot/xvla-base \
  --dataset.repo_id=Rakeshdharan56/pick_maniskill --dataset.revision=v1 --dataset.video_backend=pyav \
  --rename_map='{"observation.images.agentview": "observation.images.image", "observation.images.eye_in_hand": "observation.images.image2"}' \
  --policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --batch_size=32 --steps=7500 --save_freq=5000 --num_workers=16 --seed=1000 \
  --output_dir=outputs/train/xvla_panda_maniskill_v1 --job_name=xvla-pick_maniskill_v1-rakesh \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/xvla_panda_maniskill_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion

# X-VLA on FR3 Duo (Aravind's Newton data): 3 cameras, 16-D state/action
lerobot-train --policy.path=lerobot/xvla-base \
  --dataset.repo_id=aravindsuresh16/cathlab_fr3_newton_v1 --dataset.revision=v1.0.1 --dataset.video_backend=pyav \
  --rename_map='{"observation.images.external_camera": "observation.images.image", "observation.images.left_wrist_camera": "observation.images.image2", "observation.images.right_wrist_camera": "observation.images.image3"}' \
  --policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 \
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false \
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true \
  --batch_size=32 --steps=<from the smoke, see below> --save_freq=2500 --num_workers=16 --seed=1000 \
  --output_dir=outputs/train/xvla_fr3_pick_v1 --job_name=xvla-fr3_newton_v1.0.1-rakesh \
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/xvla_fr3_pick_v1 \
  --wandb.enable=true --wandb.project=cathlab-companion
```

**X-VLA-FR3 notes** (checked against the configs on HF):
- **Fit:** `xvla-base` has three image slots (`image`, `image2`, `image3`), `max_state_dim` 20 and `max_action_dim` 20, so our 16-D state and action fit. `action_mode=auto` reads the dataset's 16-D action, computes the loss on those 16 dims only and trims the output back to 16.
- **Camera slots:** if startup complains about cameras (the base processor also lists an `empty_camera_0`), add `--policy.empty_cameras=0` or make the smallest fix the error asks for, and log it.
- **Size:** the dataset is 384 episodes / 159,568 frames at 30 fps, 2.3× the robosuite set. **Steps:** as many as finish by ~21:30 at the smoke's steps/s, **at most 15,000**. The Panda recipe's 7,500 × 32 is the floor if time allows; below that, note it in the model card.
- **Gripper columns:** gripper is 1 = open, 0 = closed. The idle left arm's constant columns already have std = 1 and widened quantiles in `v1.0.1`; X-VLA uses MEAN_STD for actions, so that is fine.

## 6. Deadlines (freeze ≈ 23:20)

| By | What |
|---|---|
| 18:15 | T0 + T1 smokes (3 runs); first two full runs started; ETAs in `docs/STATUS.md` |
| 18:45 | T3 eval env ready; T4 adapter on `main` |
| 19:30 | T5 for the 3 robosuite-trained checkpoints + expert in both engines (first cross-physics numbers) |
| ~21:30 | `smolvla_panda_maniskill_v1`, `xvla_panda_maniskill_v1`, `xvla_fr3_pick_v1` on HF; Aravind notified about `xvla_fr3_pick_v1` |
| 22:00 | T5 for the ManiSkill checkpoints; cross-physics table final |
| 22:30 | Report back |

## 7. Report back to the master session

Post:
- per model: steps × batch, VRAM peak, wall time, final loss, HF link, W&B run;
- the cross-physics table (overall + per tier, with CIs);
- the failed-stage histogram;
- latency;
- any deviation from the v1 recipes (config diff from T1).

These feed version/promote (Sharan's registry, once it lands), the failure report and the next loop.

## 8. Rules

- Stay in `train/`, `tests/train/`, `engines/policies.py` (the `lerobot:` scheme only), `reports/robosuite/*.md|*.json`, `docs/`.
- Don't edit `contracts/`, `dataengine/`, `sim/`, the engine eval code, or `configs/variation/cathlab_v1.yaml`. Ask Rakesh.
- Secrets (HF, W&B) come from the CLI logins. Never print or commit them.
- Log every non-obvious choice in `docs/decisions.md`: batch changes, OOM fallbacks, step changes, eval deviations.
