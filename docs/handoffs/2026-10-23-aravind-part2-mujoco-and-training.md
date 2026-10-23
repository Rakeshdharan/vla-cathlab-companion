# Session Handover: Aravind, part 2: FR3 MuJoCo task data + FR3 model training (cloud GPU)

*From the master session, 2026-10-23 ~17:30 (≈ H26). Owner: **Aravind**, in his own repo [`aravinds1610/mujoco_cathlab_vla`](https://github.com/aravinds1610/mujoco_cathlab_vla). Continues [`2026-10-23-aravind-newton-fr3-session.md`](2026-10-23-aravind-newton-fr3-session.md). Paste this file into your agent as the first message, or say "fetch `docs/handoffs/2026-10-23-aravind-part2-mujoco-and-training.md` from Rakeshdharan/vla-cathlab-companion and continue".*

## 0. Feedback on part 1

Excellent work. The Newton track is the best-documented engine in the project. What we're taking from it:

- [`cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1): 384 episodes, expert 100 % on every object and every held-out tier, MuJoCo ↔ Newton sim2sim agreement 60/60. It is the team's FR3 dataset.
- The **`v1.0.1` stats fix** (widened q01/q99 on the idle left arm) is correct for any quantile-normalised model. **Train every FR3 model on `v1.0.1`.**
- The throughput finding (Newton 1,580 eps/h vs a 12-process CPU pool at 2,990 eps/h on your laptop) is honest and useful. The pitch frames Newton as *the second physics + NVIDIA stack*, not raw speed.
- Your 10 Newton workarounds (§2 of your report) are a reusable asset. Keep them in `docs/NEWTON_REPORT.md`.
- Your deviations are accepted: bottle Ø77 × 152 mm with a top grasp, mirrored tray, no extra distractors.

**Cleanup the team made in parallel (FYI, nothing for you to change):**
- **Isaac Sim is removed** from the platform. The FR3 Duo now lives only in your MuJoCo/Newton stack.
- Panda engines on `main`: robosuite (MuJoCo) and **ManiSkill3 (PhysX 5)** (`pick_maniskill`, 369 episodes paired with robosuite). Genesis is optional.
- Panda checkpoints on HF (trained on `pick_robosuite@v1`): `Rakeshdharan56/{smolvla,act,xvla}_panda_pick_v1`.
- Team status page: [`docs/STATUS.md`](https://github.com/Rakeshdharan/vla-cathlab-companion/blob/main/docs/STATUS.md). Rakesh updates it from your HF repos.

> **Model set (decided H26.5, same for Panda and FR3): ACT, SmolVLA, X-VLA, with the scripted expert as the baseline.** π0.5 and MolmoAct2 are dropped team-wide. Every table you produce has exactly these four rows.

## 1. Goal of this session (in order)

1. **Publish the FR3 MuJoCo dataset of the *team task*:** same task, objects, arm, grid and schema as the Newton set, physics = CPU MuJoCo. That gives the FR3 line a second engine's *data*, not only a second engine for checking outcomes.
2. **Train FR3 models on a cloud GPU** and push the checkpoints to HF.
3. **Evaluate them closed-loop** in Newton and CPU MuJoCo: the FR3 results table plus the FR3 sim2sim number.
4. **Retire the legacy dataset** `cathlab_handover_mujoco_v1` (left arm, 6 instruments) so nobody trains on it by mistake.

## 2. Decisions (don't re-open)

| # | Decision |
|---|---|
| B1 | **The MuJoCo dataset uses exactly the Newton task:** right arm; bottle (Ø77 × 152 mm, top grasp) / vial / syringe; your mirrored pick region and tray; `configs/variation/cathlab_fr3_newton.yaml`; `generation_order`; seed 0 = the 384-cell grid; instruction = `seed % 3`. **Not** your old 6 instruments; the old `cathlab_sim/expert.py` left-arm path is legacy |
| B2 | **Same schema as `cathlab_fr3_newton_v1`** (Siemens `fr3_duo_16d`): 16-D state/action, 3 cameras at 224², 30 fps, `success`, `robot_type fr3_duo`. Same A6 data-quality rules (no-op drop, 15-frame success hold, full attempt log with versions + git commit). Same stats fixes (std = 1 for constant columns **and** the v1.0.1 quantile widening) |
| B3 | **Dataset repo `aravindsuresh16/cathlab_fr3_mujoco_v1`**, tags `v1` + `v3.0`. Manifest `scene_id: cathlab_fr3_mujoco`, `source: scripted`. `episodes_extra.jsonl` keeps `engine: mujoco` |
| B4 | **Model repos** `aravindsuresh16/<model>_fr3_pick_v<N>`, e.g. `aravindsuresh16/smolvla_fr3_pick_v1`. Keep LeRobot's `train_config.json` (it records `dataset.repo_id` + `revision`). The model card names the dataset + tag, base checkpoint, steps × batch, GPU and wall time |
| B5 | **Training data:** v1 models on **`cathlab_fr3_newton_v1@v1.0.1`**. Only after §5 T2 shows MuJoCo data is fine, a v2 on Newton + MuJoCo combined (768 episodes) to test cross-engine co-training |
| B6 | **Eval protocol** = the team's: `eval_configs(spec, tier, seeds)` over all 5 tiers × 3 objects; **seeds 5000–5003** (`configs/eval/suite_v2.yaml`: 60 episodes per model per engine); Wilson 95 % CI; failed-stage histogram; latency p50/p95. Use your geometric stage predicates (`task_stages.py`) so both engines are judged identically |

## 3. What to do: data

**D1: Generate (CPU MuJoCo, your runner already supports it)**

```bash
source .venv-newton/bin/activate      # runner + cpu_sim live here; MuJoCo 3.12 CPU
python -m cathlab_sim.newton.runner generate --engine mujoco --out datasets/mujoco_raw/v1 --count 384 --seed 0 --procs 12
```

- ✅ 384 attempts; expert ≥ 95 % per object (your sim2sim run says it should be 100 %). Failures logged in `meta/episodes_extra.jsonl`.
- Record episodes/hour (your report measured ~2,990 eps/h with 12 processes).

**D2: Convert + publish** (LeRobot venv)

```bash
source .venv/bin/activate
python -m cathlab_sim.newton.convert --raw datasets/mujoco_raw/v1 --out datasets/cathlab_fr3_mujoco_v1 \
  --repo-id aravindsuresh16/cathlab_fr3_mujoco_v1 --push --tag v1
```

- `convert.py` currently hard-codes `scene_id="cathlab_fr3_newton"` and the Newton dataset tags/notes. Make them follow the raw data's `engine` field (`cathlab_fr3_mujoco`, tags `fr3_duo`, `mujoco`), then re-check the Newton conversion is unchanged.
- Apply the quantile widening (`floor_constant_stats`) **before** the first push, so `v1` is already right and needs no `v1.0.1`.
- ✅ `LeRobotDataset("aravindsuresh16/cathlab_fr3_mujoco_v1", revision="v1")` loads: `action (16,)`, `state (16,)`, 3 × `(3, 224, 224)`, fps 30, `fr3_duo`. Preview GIF in the repo. Episode *i* has the same config as Newton episode *i* (check a few `episodes_extra` rows side by side).

**D3: Retire the legacy dataset.** Put a banner at the top of `aravindsuresh16/cathlab_handover_mujoco_v1`'s README: "Legacy (pre-task): left arm, 6 instruments. For the team task use `cathlab_fr3_newton_v1` / `cathlab_fr3_mujoco_v1`." Same note in your repo README. Don't delete it: it is referenced in history.

## 4. What to do: training (cloud GPU)

**Setup:** one Python 3.12 venv with your LeRobot version (0.6.1, the one that wrote the datasets) plus the extras for the models you train, e.g. `pip install "lerobot[smolvla,xvla]==0.6.1"`. Check `torch.cuda.is_available()` and the GPU name first.
- Log in to HF (`hf auth login`) and W&B (project `cathlab-companion`, run name `<model>-fr3_newton_v1.0.1-aravind`). Keys go in env vars, never in git.
- If the instance has a Blackwell GPU (RTX 50xx / RTX PRO 6000 / B200), torch must be built for CUDA ≥ 12.8.

**Model plan** (in this order; stop when the budget runs out):

| # | Model | Base | Why | Starting recipe |
|---|---|---|---|---|
| M1 | **SmolVLA** | `lerobot/smolvla_base` | Small and fast. Its three camera slots match ours. Your adapter `smolvla_policy.py` already exists. Your v1 run was undertrained (3k steps, 0/60); this fixes that | `--rename_map` external→`camera1`, left_wrist→`camera2`, right_wrist→`camera3` (as in your README). **≥ 1 full pass:** 384 episodes ≈ 160k frames → batch 64 × ≥ 2,500 steps; aim for **15–20k steps**. 16-D state/action pad into its 32 slots |
| M2 | **X-VLA** (0.9B) | `lerobot/xvla-base` | Frontier model, the same one the Panda line uses. Its base checkpoint has three image slots (`observation.images.image`, `image2`, `image3`). With `action_mode=auto` it reads our 16-D action and pads it to its 20-D head (16 ≤ 20) | `--rename_map` external→`image`, left_wrist→`image2`, right_wrist→`image3`; `--policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20 --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false --policy.train_policy_transformer=true --policy.train_soft_prompts=true`; batch 32, ≤ 20k steps. The Panda run needed 7,500 steps × 32. **Smoke first:** if a camera slot mismatches (`image3` vs an `empty_camera_0`), adjust `--policy.empty_cameras` and log it. Assert the postprocessed action is `(*, 16)` |
| M3 | **ACT** | from scratch | Non-language lower bound. With 3 objects present it can't know the target from the instruction, so expect wrong-object failures. That's the point | `--policy.type=act --policy.chunk_size=50 --policy.n_action_steps=50`, batch 64, 50k steps |

- **Flags differ between LeRobot versions.** The Panda recipes in our [training handoff](2026-10-23-training-gpu-session.md) were verified on LeRobot 0.5.2 (`b06ad40`). On 0.6.1, confirm every flag with `lerobot-train --help` and log the deviation.
- **Always smoke first:** 200 steps, `--policy.push_to_hub=false`, separate `--output_dir`. Check that a checkpoint is saved and the postprocessed action is `(*, 16)`.
- Push each finished model to `aravindsuresh16/<model>_fr3_pick_v1` (B4).
- ✅ Per model: the HF repo loads; W&B shows loss falling. Record steps × batch, VRAM peak, wall time and final loss.

## 5. What to do: evaluation (the FR3 results)

**T1: Closed-loop eval in Newton and CPU MuJoCo**
- Wrap each checkpoint as a `PolicyAdapter`. Generalise `smolvla_policy.py` into one LeRobot adapter that resolves the policy class from the checkpoint config: `PreTrainedConfig.from_pretrained` → `lerobot.policies.factory.get_policy_class(cfg.type)` → `make_pre_post_processors(...)` → `predict_action_chunk` → postprocess.
- Execute the first `n_exec` actions (start with 25), then re-plan.
- Run B6's protocol (5 tiers × 3 objects × seeds 5000–5003 = 60 episodes) **in both engines**, with the same seeds and the same layouts.
- ✅ `reports/eval_<model>_<engine>.json` per model and engine: success by tier × object with Wilson CIs, failed-stage histogram, latency. The expert row is the reference (100 % from your report).

**T2: The FR3 tables**

Write these into `docs/NEWTON_REPORT.md` §9 (new):

| | in_dist | held-out layout | held-out obj colour | held-out table colour | all held-out | overall (CI) |
|---|---|---|---|---|---|---|
| expert (Newton) | … | | | | | |
| SmolVLA (Newton) | … | | | | | |
| SmolVLA (MuJoCo) | … | | | | | |
| X-VLA (Newton) / (MuJoCo) | … | | | | | |
| ACT (Newton) / (MuJoCo) | … | | | | | |

Plus **the FR3 sim2sim gap:** the same checkpoint's success in Newton vs MuJoCo, per tier, and the rank agreement of the models across the two engines.

**T3 (only if time is left): v2 co-training.** Train SmolVLA on Newton + MuJoCo combined (768 episodes, same steps), then evaluate as in T1. Does training on both engines shrink the Newton↔MuJoCo gap or lift the held-out tiers? One line in the report either way.

## 6. Deadlines (freeze ≈ 23:20)

| By | What |
|---|---|
| 18:30 | D1 + D2: `cathlab_fr3_mujoco_v1` on HF; D3 banner |
| 19:00 | M1 SmolVLA training on the cloud GPU (smoke done) |
| 20:30 | SmolVLA on HF; T1 eval for SmolVLA in both engines; message Rakesh with the numbers |
| 22:00 | M2 (X-VLA) and M3 (ACT) on HF and evaluated if the budget allows; T2 tables in `NEWTON_REPORT.md` |
| 22:30 | Final message to Rakesh: dataset + model links, the tables, cloud GPU type and cost |

## 7. Report back

Send Rakesh the HF links, the T2 tables, and any deviations, especially LeRobot 0.6.1 flag differences. He adds them to `docs/STATUS.md` and the pitch.

## 8. Rules

- Commit as yourself in your repo, with no AI `Co-authored-by` trailers. Datasets, videos and checkpoints never go into git.
- Cloud credentials and HF/W&B tokens live only in env vars on the instance. **Shut the instance down when training is done.**
- Don't change the schema (B2) or the task (B1). Ask Rakesh if something blocks you.
- Keep the `CREDITS` lines (Newton, MuJoCo/MuJoCo-Warp, Menagerie, LeRobot/SmolVLA/X-VLA licences).
