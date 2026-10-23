# Rakesh — Lead: Shared Sim Core, robosuite Track, Training, Loop, Integration

> **Update (2026-10-23 H24): Isaac Sim is dropped** (no working run after a day). Engines now: robosuite and ManiSkill3 (Panda), Genesis planned, FR3 Duo via Aravind's MuJoCo/Newton track. Isaac items below are historical.

> **Updated H6.5.** Single-arm Franka Panda, our own table scene in Isaac Sim + robosuite, LIBERO-compatible I/O. See the [simulation framework spec](../specs/2026-10-22-sim-framework-design.md).

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| R1 | Foundation contracts on `main` (merge the reviewed `feat/akanksh-eval-harness` contracts or plan Tasks 1–4) | `contracts/`, `eval/runner.py` | H7 |
| R2 | **Shared sim core**: [plan](../plans/2026-10-22-sim-shared-core.md) Tasks 1–6 (variation spec, expert, fake sim, generator, LeRobot sink) | `dataengine/`, `configs/variation/`, `configs/tasks/cathlab_pick.yaml` | H8 |
| R3 | **robosuite env** `CathlabPickPlace` (from `PickPlace`): Panda + `OSC_POSE`, table (top at base z = 0 after offset), tray, 3 objects at spec dimensions, `agentview` + `robot0_eye_in_hand` at 256², 20 Hz; randomizer applying `EpisodeConfig` | `sim/robosuite/env.py`, `sim/robosuite/randomize.py` | H10 |
| R4 | `RobosuiteExpertSim` (`ExpertSim`) + `RobosuiteBackend` (`SimBackend`); expert ≥80% on 20 train configs | `sim/robosuite/expert_sim.py`, `sim/robosuite/backend.py` | H12 |
| R5 | ✅ 384 train episodes in parallel shards → [`Rakeshdharan56/pick_robosuite@v1`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) | data on HF | H16 |
| R6 | 01 Ingest: provided FR3 Duo datasets validated + reported (embodiment-agnostic demo); instruction paraphrases for the 3 objects | `ingest/` | H14 |
| R7 | **Model set: ACT, SmolVLA, X-VLA + expert baseline.** ✅ all three on `pick_robosuite@v1` (HF). ⏳ all three on `pick_maniskill@v1` (RTX 6000 Ada, [handoff](../handoffs/2026-10-23-maniskill-training-session.md)) → adapter → cross-physics eval (robosuite ↔ ManiSkill) → version/promote → Panda v2 co-training. FR3: Aravind. [Status](../STATUS.md) | `train/` | H18–H28 |
| R8 | Loop: failing grid cells → `eval_configs`-style configs for those cells → generate in both sims → v2 | `loop/` | H28 |
| R9 *(stretch)* | RoboSimStudio: prompt → cathlab scene showcase; optional held-out eval with a Panda embodiment | `sim/robosimstudio/` | H30 |

## Notes

- Share the robosuite `agentview` camera pose with Mitesh by H8, so both sims render the same viewpoint.
- robosuite `OSC_POSE` takes the normalized action unchanged (output_max 0.05 m / 0.5 rad), so no scaling is needed.
- Data versions:
  - v0 = `pick_isaac@v1`;
  - v1 = v0 + `pick_robosuite@v1`;
  - v2 = v1 + loop-targeted episodes.
