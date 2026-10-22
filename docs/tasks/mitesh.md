# Mitesh — Isaac Sim Track (Isaac Lab): Scene, Randomizer, Expert Adapter, Data

> **Updated H6.5.** The robot is now a single-arm **Franka Panda** in **our own simple table scene**. The FR3 Duo and the provided USD scene are out. Read the [simulation framework spec](../specs/2026-10-22-sim-framework-design.md) and the [shared-core plan](../plans/2026-10-22-sim-shared-core.md) first.

## Agent brief (paste to your agent)

> You are building the Isaac Sim track of the CathLab Companion pipeline (spec: `docs/specs/2026-10-22-sim-framework-design.md`; shared core: `docs/plans/2026-10-22-sim-shared-core.md`, package `dataengine/`).
>
> Task: in Isaac Lab, create a manager-based environment derived from the Franka lift/stack tasks:
> - Franka Panda + Franka hand;
> - a table (top at robot-base-frame z = 0);
> - a tray, inner 0.25 × 0.32 m with 0.04 m walls, centred at `scene.tray_center`;
> - three primitive-composite rigid objects with the exact dimensions in `configs/variation/cathlab_v1.yaml`: bottle (cylinder Ø62 × 250 mm), vial (Ø27 × 42 mm), syringe (barrel Ø15 × 100 mm + plunger Ø8 × 40 mm + flange, lying flat);
> - two cameras, `agentview` (front, third person) and `eye_in_hand` (on `panda_hand`), 256×256, 20 Hz control.
>
> At reset, apply an `EpisodeConfig`: object poses from `dataengine.variation.object_poses`, object/table colours from the palettes, mass × `mass_scale`, `friction`, and light intensity / colour temperature.
>
> Implement `IsaacExpertSim` (the `dataengine.expert.ExpertSim` protocol):
> - poses in the robot base frame;
> - TCP pose of the fingertip centre;
> - normalized 7-D actions mapped to `DifferentialIKController` relative-pose commands (×0.05 m, ×0.5 rad) plus a binary gripper;
> - `tray_contains` true when the object is at rest inside the tray inner bounds.
>
> Also implement `IsaacBackend` (`contracts.SimBackend`) with stage flags reach/grasp/lift/transport/place. Then run `dataengine.generate.generate_scripted(..., source_sim="isaac")` with `LeRobotSink`. Unit tests must not need Isaac (mark real-sim tests `@pytest.mark.gpu`).

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| I1 | Isaac Lab env loads: Panda, table, tray, 3 objects, 2 cameras, renders at 256² | `sim/isaac/env_cfg.py`, `sim/isaac/scene.py` | H8 |
| I2 | Randomizer applying an `EpisodeConfig` (poses, colours, mass, friction, lights); screenshot grid of 8 configs | `sim/isaac/randomize.py` | H10 |
| I3 | `IsaacExpertSim` + `IsaacBackend`; `run_expert_episode` reaches ≥80% success on 20 train configs | `sim/isaac/expert_sim.py`, `sim/isaac/backend.py` | H12 |
| I4 | 384 train episodes → `cathlab/pick_isaac@v1` (LeRobot v3 + `manifest.yaml` + `meta/episodes_extra.jsonl`), uploaded to HF | data on HF | H16 |
| I5 | Throughput note (episodes/min/GPU) + per-object expert success in `docs/decisions.md` | — | H16 |
| I6 *(stretch)* | Cosmos Transfer on ~50 episodes → `cathlab/pick_isaac_cosmos@v1` | — | H30 |
| I7 *(stretch)* | GraspGen "expert v2" grasp ranking in Isaac | — | after I4 |

## Interfaces

- **Consumes:**
  - `dataengine.variation` (`VariationSpec`, `EpisodeConfig`, `object_poses`, `train_configs`, `eval_configs`);
  - `dataengine.expert` (`ExpertSim`, `run_expert_episode`, `Q_DOWN`);
  - `dataengine.generate.generate_scripted`;
  - `dataengine.lerobot_sink.LeRobotSink`;
  - `contracts.SimBackend`, `StepInfo`.
- **Produces:** `IsaacExpertSim`, `IsaacBackend`, dataset `cathlab/pick_isaac@v1`.

## Acceptance

- The same `EpisodeConfig` gives the same object placement as robosuite (compare `get_object_pose` outputs).
- Expert ≥80% success per object; failures are logged by waypoint (`ExpertResult.trace`).
- The dataset loads with `LeRobotDataset` and has exactly the schema from `dataengine.lerobot_sink.panda_libero_features()`.

## Notes

- **Quaternion convention:** `Q_DOWN = [0, 1, 0, 0]` (180° about x) assumes the EEF z-axis is the approach axis. If Isaac's `panda_hand` frame differs, pass the correct `q_down`.
- **Camera match:** keep the `agentview` pose numerically identical to robosuite's (Rakesh shares it at H8).
