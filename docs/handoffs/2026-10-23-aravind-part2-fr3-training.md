# Session Handover: Aravind, part 2: FR3 SmolVLA on the Newton dataset (cloud GPU) + FR3 eval

*From the master session, 2026-10-23 ~17:50 (≈ H26.5). Owner: **Aravind**, in his own repo [`aravinds1610/mujoco_cathlab_vla`](https://github.com/aravinds1610/mujoco_cathlab_vla). Continues [`2026-10-23-aravind-newton-fr3-session.md`](2026-10-23-aravind-newton-fr3-session.md). Paste this file into your agent as the first message, or say "fetch `docs/handoffs/2026-10-23-aravind-part2-fr3-training.md` from Rakeshdharan/vla-cathlab-companion and continue".*

## 0. Feedback on part 1

Excellent work. The Newton track is the best-documented engine in the project. What we're taking from it:

- The **`v1.0.1` stats fix** (widened q01/q99 on the idle left arm) is correct for any quantile-normalised model. **Train every FR3 model on `v1.0.1`.**
- The throughput finding (Newton 1,580 eps/h vs a 12-process CPU pool at 2,990 eps/h on your laptop) is honest and useful. The pitch frames Newton as *the second physics + NVIDIA stack*, not raw speed.
- Your 10 Newton workarounds (§2 of your report) are a reusable asset. Keep them in `docs/NEWTON_REPORT.md`.
- Your deviations are accepted: bottle Ø77 × 152 mm with a top grasp, mirrored tray, no extra distractors.

**Your datasets, and which one we use:**

| Dataset | Use |
|---|---|
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) | ✅ **The team's FR3 dataset.** 384 episodes, our task (right arm, bottle/vial/syringe), expert 100 % on every tier. **Train on tag `v1.0.1`** |
| `aravindsuresh16/cathlab_handover_mujoco_v1` | ❌ Not used: the old setup (left arm, 6 instruments). Out of the team's records; don't train or evaluate on it. No action needed |

**No new dataset in this session.** Train only on the Newton dataset.

**Cleanup the team made in parallel (FYI, nothing for you to change):**
- **Isaac Sim is removed** from the platform. The FR3 Duo now lives only in your Newton/MuJoCo stack.
- Panda engines on `main`: robosuite (MuJoCo) and **ManiSkill3 (PhysX 5)**, with `pick_maniskill` (369 episodes paired with robosuite). Genesis is optional.
- Panda checkpoints on HF (trained on `pick_robosuite@v1`): `Rakeshdharan56/{smolvla,act,xvla}_panda_pick_v1`. SmolVLA and X-VLA are now training on ManiSkill data.
- Team status page: [`docs/STATUS.md`](https://github.com/Rakeshdharan/vla-cathlab-companion/blob/main/docs/STATUS.md). Rakesh updates it from your HF repos.

> **FR3 model set (decided 17:55): SmolVLA and X-VLA, with the scripted expert as the baseline. ACT is dropped for FR3.** π0.5 and MolmoAct2 are dropped team-wide.
>
> | FR3 model | Trained by | Where | Checkpoint | ETA |
> |---|---|---|---|---|
> | **SmolVLA** | **you** | cloud GPU | `aravindsuresh16/smolvla_fr3_pick_v1` | **~20:30** |
> | **X-VLA** | Rakesh | RTX 6000 Ada | `Rakeshdharan56/xvla_fr3_pick_v1` (same dataset: `cathlab_fr3_newton_v1@v1.0.1`) | ~21:30 |
>
> **You evaluate both** in Newton + CPU MuJoCo (§4). Every FR3 table has three rows: expert, SmolVLA, X-VLA.

## 1. Goal of this session (in order)

1. **Train SmolVLA** on `cathlab_fr3_newton_v1@v1.0.1` on a cloud GPU and push it to HF by ~20:30. X-VLA is trained by Rakesh; ACT is dropped.
2. **Evaluate SmolVLA and Rakesh's X-VLA closed-loop in Newton**, then in CPU MuJoCo as a second physics engine. That is **evaluation only**: your `cpu_sim.py` twin already runs the same scene, so no MuJoCo dataset is needed. The result is the FR3 results table plus the FR3 sim2sim number.

## 2. Decisions (don't re-open)

| # | Decision |
|---|---|
| B1 | **Training data:** `cathlab_fr3_newton_v1` at tag **`v1.0.1`**, nothing else |
| B2 | **Model repos** `aravindsuresh16/<model>_fr3_pick_v<N>`, e.g. `aravindsuresh16/smolvla_fr3_pick_v1`. Keep LeRobot's `train_config.json` (it records `dataset.repo_id` + `revision`). The model card names the dataset + tag, base checkpoint, steps × batch, GPU and wall time |
| B3 | **Eval protocol** = the team's: `eval_configs(spec, tier, seeds)` over all 5 tiers × 3 objects; **seeds 5000–5003** (`configs/eval/suite_v2.yaml`: 60 episodes per model per engine); Wilson 95 % CI; failed-stage histogram; latency p50/p95. Use your geometric stage predicates (`task_stages.py`) so both engines are judged identically |

## 3. What to do: training (cloud GPU)

**Setup:** one Python 3.12 venv with your LeRobot version (0.6.1, the one that wrote the datasets) plus the extras for the models you train, e.g. `pip install "lerobot[smolvla,xvla]==0.6.1"` (`xvla` too, because you evaluate Rakesh's X-VLA). Check `torch.cuda.is_available()` and the GPU name first.
- Log in to HF (`hf auth login`) and W&B (project `cathlab-companion`, run name `<model>-fr3_newton_v1.0.1-aravind`). Keys go in env vars, never in git.
- If the instance has a Blackwell GPU (RTX 50xx / RTX PRO 6000 / B200), torch must be built for CUDA ≥ 12.8.

**Model plan:**

| # | Model | Base | Why | Starting recipe |
|---|---|---|---|---|
| M1 | **SmolVLA** | `lerobot/smolvla_base` | Small and fast. Its three camera slots match ours. Your adapter `smolvla_policy.py` already exists. Your v1 run was undertrained (3k steps, 0/60); this fixes that | `--rename_map` external→`camera1`, left_wrist→`camera2`, right_wrist→`camera3` (as in your README). **≥ 1 full pass:** 384 episodes ≈ 160k frames → batch 64 × ≥ 2,500 steps; aim for **15–20k steps**. 16-D state/action pad into its 32 slots |
| M2 | **X-VLA** (0.9B) | `lerobot/xvla-base` | **Trained by Rakesh on the RTX 6000 Ada**, not by you. Same dataset and tag. Checkpoint `Rakeshdharan56/xvla_fr3_pick_v1` (cameras external→`image`, left_wrist→`image2`, right_wrist→`image3`; `action_mode=auto`, 16-D) | Nothing to train. Evaluate it in §4 when Rakesh says it's on HF (~21:30) |

- **Flags differ between LeRobot versions.** The Panda recipes in our [training handoff](2026-10-23-training-gpu-session.md) were verified on LeRobot 0.5.2 (`b06ad40`). On 0.6.1, confirm every flag with `lerobot-train --help` and log the deviation.
- **Always smoke first:** 200 steps, `--policy.push_to_hub=false`, separate `--output_dir`. Check that a checkpoint is saved and the postprocessed action is `(*, 16)`.
- Push SmolVLA to `aravindsuresh16/smolvla_fr3_pick_v1` (B2).
- ✅ Per model: the HF repo loads; W&B shows loss falling. Record steps × batch, VRAM peak, wall time and final loss.

## 4. What to do: evaluation (the FR3 results)

**T1: Closed-loop eval in Newton and CPU MuJoCo**
- Wrap each checkpoint as a `PolicyAdapter`. Generalise `smolvla_policy.py` into one LeRobot adapter that resolves the policy class from the checkpoint config: `PreTrainedConfig.from_pretrained` → `lerobot.policies.factory.get_policy_class(cfg.type)` → `make_pre_post_processors(...)` → `predict_action_chunk` → postprocess.
- Feed the **dataset key names** (`observation.images.external_camera`, …), as `smolvla_policy.py` already does. Each checkpoint's saved preprocessor contains the camera rename used in training (SmolVLA → `camera1–3`, X-VLA → `image`/`image2`/`image3`), so one adapter serves both models.
- Execute the first `n_exec` actions (start with 25), then re-plan.
- Run B3's protocol (5 tiers × 3 objects × seeds 5000–5003 = 60 episodes) **in both engines**, with the same seeds and the same layouts.
- ✅ `reports/eval_<model>_<engine>.json` per model and engine: success by tier × object with Wilson CIs, failed-stage histogram, latency. The expert row is the reference (100 % from your report).

**T2: The FR3 tables**

Write these into `docs/NEWTON_REPORT.md` §9 (new):

| | in_dist | held-out layout | held-out obj colour | held-out table colour | all held-out | overall (CI) |
|---|---|---|---|---|---|---|
| expert (Newton) | … | | | | | |
| SmolVLA (Newton) | … | | | | | |
| SmolVLA (MuJoCo) | … | | | | | |
| X-VLA (Newton) / (MuJoCo) | … | | | | | |

Plus **the FR3 sim2sim gap:** the same checkpoint's success in Newton vs MuJoCo, per tier, and the rank agreement of the models across the two engines.

## 5. Deadlines (freeze ≈ 23:20)

| By | What |
|---|---|
| now | SmolVLA training on the cloud GPU (smoke done) |
| 20:30 | `smolvla_fr3_pick_v1` on HF; message Rakesh |
| 21:00 | §4 T1 eval of SmolVLA + expert in Newton and MuJoCo; message Rakesh with the numbers |
| ~21:30 | Rakesh's `xvla_fr3_pick_v1` lands on HF → evaluate it the same way |
| 22:15 | T2 FR3 tables (expert, SmolVLA, X-VLA) in `NEWTON_REPORT.md` |
| 22:30 | Final message to Rakesh: model link, the tables, cloud GPU type and cost. **Shut the cloud instance down** |

## 6. Report back

Send Rakesh the HF links, the T2 tables, and any deviations, especially LeRobot 0.6.1 flag differences. He adds them to `docs/STATUS.md` and the pitch.

## 7. Rules

- Commit as yourself in your repo, with no AI `Co-authored-by` trailers. Datasets, videos and checkpoints never go into git.
- Cloud credentials and HF/W&B tokens live only in env vars on the instance. **Shut the instance down when training is done.**
- Don't change the dataset or the task. Ask Rakesh if something blocks you.
- Keep the `CREDITS` lines (Newton, MuJoCo/MuJoCo-Warp, Menagerie, LeRobot/SmolVLA/X-VLA licences).
