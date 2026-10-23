# Session Handover: Isaac Sim Track (Provided FR3 Duo Scene) → Objects, Expert, Recording, Variation

> **Superseded (2026-10-23):** follow [`2026-10-23-isaac-engine-session.md`](2026-10-23-isaac-engine-session.md): Isaac as an engine plugin with the uniform CLI. Keep this file for the verified USD facts in §1.

*From the master session, 2026-10-22 ~22:15 (≈H8). For the whole team and your coding agents; Rakesh is helping. Paste this file into your agent as the first message, or say "follow docs/handoffs/2026-10-22-isaac-fr3-session.md".*

## Goal

In the **provided Isaac Sim FR3 Duo hospital scene** (`Hackathon_FR3Duo_flattened.usd`), build a scripted data factory for **"Hand me the {bottle | vial | syringe}"**:

1. add and prepare our 3 objects;
2. add a right-arm scripted expert that picks the named object and places it on the serving tray;
3. record episodes in **exactly the provided Siemens dataset schema**;
4. randomize scene variations;
5. generate synthetic data at scale (Isaac Replicator randomizers; Cosmos and Mimic as stretch goals).

A parallel session builds the same task in **robosuite with a Panda** (Rakesh). Don't touch `sim/robosuite/`.

## Decisions (made with Rakesh; don't re-open)

| # | Decision |
|---|---|
| 1 | **Scene:** the provided `Hackathon_FR3Duo_flattened.usd`, unchanged except for our objects. Keep the hospital room: it's our realism advantage |
| 2 | **Robot:** FR3 Duo as provided. **Single-arm: the RIGHT arm does pick and place.** The left arm holds its ready pose but is still recorded in the 16-D action/state |
| 3 | **Task:** all 3 objects are on a pick zone on the **right supply cart** (`SM_SupplyCart_01e`) beside the serving tray. The robot picks the named one and places it in `ServingTray` |
| 4 | **Objects:** bottle = the scene's `NaturalBostonRoundBottle_A02_PR_NVD_01`; vial = the scene's `SM_PillBottle_01v`; syringe = a **new** primitive composite (barrel Ø15 × 100 mm + plunger Ø8 × 40 mm + flange 30 × 10 × 3 mm, lying flat, ~0.015 kg) |
| 5 | **Data schema = the provided Siemens dataset** (same as `Rakeshdharan56/fr3_duo_handover_84ep`), so our episodes can be co-trained with the 94 provided demos. See §3 |
| 6 | **Expert:** privileged object pose → grasp template → waypoints → right-arm IK → absolute joint targets. GraspGen comes later |
| 7 | **Variation grid:** same as the robosuite track: 3 objects × 8 layouts × 4 object-colour palettes × 4 cart-top colours = **384 train episodes**. Held out: layouts 8–9, palette 4, cart colour 4. Mass ×U(0.7, 1.3), friction U(0.5, 1.0), lighting as in the provided data (§5) |

## 1. Verified facts about the provided USD (inspected 22:10)

**Stage:**
- **Y-up**, metres; default prim `/World`; physics scene `/World/PhysicsScene` (PhysX TGS, 60 Hz in the provided recordings);
- lights `/World/RectLight` and `/World/DomeLight`.

| Prim | World bbox (x, y = up, z) | Notes |
|---|---|---|
| `/World/FR3Duo_withV_Mkit__1_` | x −4.749…−3.249, y 0.956…1.666, z 22.750…23.472 | Two articulations: `left_arm_root_joint`, `right_arm_root_joint`. TCP frames: `.../left_tcp` (−4.274, 1.403, 22.890) and `.../right_tcp` (−3.734, 1.403, 22.904) |
| `/World/ServingTray` | x −3.623…−3.374, y 0.966…1.013, z 22.543…22.889 | Place target, inner ≈ 229 × 307 mm, static colliders |
| `/World/SM_SupplyCart_01e` | x −4.118…−3.013, y −0.010…1.103, z 22.431…22.964 | Right cart: tray sits on it. **Our pick zone goes here, left of the tray** |
| `/World/SM_SupplyCart_03a` | x −5.217…−4.085, z 22.470…23.103 | Left cart: currently holds the box/bottle/pill (top ≈ y 0.82) |
| `/World/NaturalBostonRoundBottle_A02_PR_NVD_01` | size 77 × 152 × 77 mm | Dynamic rigid body, mass attr 0 → **set an explicit mass (0.30 kg)**. Our **bottle** |
| `/World/SM_PillBottle_01v` | size 27 × 42 × 27 mm | Dynamic, mass 0 → **set 0.02 kg**. Our **vial** |
| `/World/WhiteCorrugatedBox_A34_16x16x10cm_PR_NVD_01` | 31 × 72 × 107 mm | The provided task object. **Park it** (move it off the work area, or disable its rigid body) |
| Other dynamic props | `SM_Computer_02b`, `SM_BooksSet_26`, `SM_BottleA` (far away, x ≈ 1.1) | Leave as they are |

**Cameras** (already in the USD; the provided recorder used them at 224×224):
- head: `/World/FR3Duo_withV_Mkit__1_/mount_link/cover_link/head_link/zed_mini_camera_link/zed_mini_left_camera_frame/head_cam`;
- left wrist: `/World/FR3Duo_withV_Mkit__1_/left_d405_color_frame/left_wrist_cam`;
- right wrist: `/World/FR3Duo_withV_Mkit__1_/right_d405_color_frame/right_wrist_cam`.

**Right-arm joints:**

| Joint | Limits (deg) |
|---|---|
| `right_fr3v2_joint1` | ±166.2 |
| `joint2` | ±105.2 |
| `joint3` | ±166.2 |
| `joint4` | −176.3…−6.7 |
| `joint5` | ±164.8 |
| `joint6` | 25.2…264.8 |
| `joint7` | ±174.8 |

- Gripper drive: `right_gripper_ctrl`, 0…45.84° (0…0.8 rad). The 5 Robotiq mimic joints follow it.
- Ready pose (both arms) ≈ `[0, −45, 0, −135, 0, 90, 45]` deg.
- Gripper gap: 0 rad = 85 mm open, 0.8 rad = closed; hold ≈ 0.459 rad.

## 2. Tasks (suggested owners; adjust in the team chat)

Everyone keeps their main workstream; this is the shared Isaac push. Each task ends with a check (✅).

**I0 — Setup and probe** (Navneet, with Rakesh)
- Record the Isaac Sim version (4.5 uses `omni.isaac.*`, 5.x uses `isaacsim.*`) in `docs/decisions.md`.
- Load the USD **headless** (`SimulationApp({"headless": True})`), step 100 frames, save one image from each of the 3 cameras.
- **Reach probe:** with differential IK, move `right_tcp` to the 4 corners of a candidate pick zone on cart 01e and to the tray centre, at a grasp height of 5 cm above the cart top and with the gripper pointing down.
- ✅ Images saved, the pick zone fixed (the corners all reachable within 1 cm), and the cart-top height measured (expected ≈ y 0.966).

**I1 — Objects** (Navneet)
- Bottle and vial: keep the scene prims. Set explicit mass, a shared `PhysicsMaterial` (friction is varied later) and convex-hull colliders.
- Syringe: author a primitive composite under `/World/Cathlab/Syringe` (rigid body, convex colliders, 0.015 kg).
- Park the box.
- Write `sim/isaac/objects.py` with `setup_objects(stage) -> dict[name, prim_path]`.
- ✅ All 3 rest stably on the cart for 300 steps (no drift > 2 mm, no jitter).

**I2 — Task frame and layouts** (Akanksh)
- Define a **task frame** on the cart top: origin at the pick-zone centre on the cart surface, z = world +Y (up), x and y in the cart plane. Then the shared variation code works unchanged.
- Write `configs/variation/cathlab_fr3_isaac.yaml` with the **same schema** as `configs/variation/cathlab_v1.yaml`, but with this scene's geometry:
  - tray centre in the task frame;
  - pick region from I0;
  - bottle radius 0.0385, length 0.152;
  - vial radius 0.0135, length 0.042;
  - syringe as in `cathlab_v1`.
- Write `sim/isaac/frames.py` with `task_to_world(pose7)` and `world_to_task(pose7)`; quaternions are **wxyz**, as in USD (`Gf.Quatd`).
- ✅ Test: a world→task→world round-trip is exact, and the `object_poses(spec, cfg)` layouts land inside the pick zone and don't overlap the tray.

**I3 — Expert** (Navneet lead, Rakesh helps)
- Right-arm pick and place:
  1. waypoints from the shared `dataengine.expert.plan_waypoints(spec, target, obj_pose)` (target TCP poses in the task frame: pre_grasp → grasp → close → lift → above_tray → lower → open → retreat);
  2. convert to world;
  3. right-arm **differential IK** (damped least squares on the articulation Jacobian; Isaac Lab's `DifferentialIKController` or 10 lines of numpy);
  4. absolute joint targets at 60 Hz;
  5. gripper `right_gripper_ctrl` → 0 open / 0.8 rad closed.
- Measure the "gripper pointing down" quaternion of `right_tcp` once and pass it as `q_down` (don't edit the shared code).
- Success = target at rest inside the tray bounds and gripper open.
- ✅ **≥80% success per object** on 20 train configs, with the failing waypoint logged.

**I4 — Recorder in the provided schema** (Sharan)
- Write `sim/isaac/recorder.py`: physics at 60 Hz, record every 2nd tick → 30 fps.
- The features must equal the provided `meta/info.json`:
  - `observation.state` and `action`: `float32[16]` = left j1–7, left gripper, right j1–7, right gripper; joints in rad;
  - **gripper normalized 1 = open, 0 = closed** (`norm = 1 − q/0.8`);
  - `action` = **absolute joint targets**;
  - `observation.images.external_camera` (head), `observation.images.left_wrist_camera`, `observation.images.right_wrist_camera`, all 224×224 RGB;
  - `success: bool`;
  - `robot_type: fr3_duo`;
  - `task` = the instruction string ("Hand me the vial", …).
- Also write `meta/episodes_extra.jsonl` (the `EpisodeConfig`, seed, success, final placement metrics, `source_sim: isaac`) and `manifest.yaml` (`contracts.DatasetManifest`, `source: scripted`).
- Writing works with the LeRobot API: `LeRobotDataset.create(...)`, `add_frame`, `save_episode`, `finalize`.
- ✅ 5 episodes load with `LeRobotDataset`, and their feature dict equals the provided dataset's (except `raw_*_row`).

**I5 — Variation** (Navneet, after I1)
- Write `sim/isaac/randomize.py` with `apply_episode_config(stage, spec, cfg)`:
  - object poses from the layout;
  - object colours from palette `cfg.obj_color_id` (override the material's diffuse colour);
  - cart-top colour `cfg.table_color_id`;
  - mass × `mass_scale`;
  - friction;
  - `RectLight`/`DomeLight` intensity and colour temperature.
- Use `omni.replicator.core` randomizers or direct USD attribute edits, whichever is reliable per reset.
- Lighting ranges, matching the provided data:
  - colour temperature 4,393–7,905 K;
  - rect intensity 1,886–4,066;
  - dome 350–613;
  - rect tilt ±13°.
- ✅ A PNG grid of 8 configs that look visibly different, and the expert still reaches ≥80% on 20 mixed configs.

**I6 — Generate** (all)
- Write `scripts/isaac_generate.py --split train --start 0 --n 48 --out data/pick_isaac --shard i --num-shards k`, running headless.
- Split the 384 configs across machines and GPUs, merge the shards, and upload to HF as `cathlab/pick_isaac_fr3@v1` (ask Rakesh for the HF org).
- ✅ 384 episodes, with the expert success rate and throughput (episodes/min/GPU) logged in `docs/decisions.md`.

**Stretch (after I6):**
- I7: Cosmos Transfer on about 50 episodes (photoreal augmentation).
- I8: Isaac Lab Mimic re-targeting.
- I9: GraspGen grasp candidates as expert v2.

## 3. Interfaces with the rest of the pipeline

- **Shared code you reuse** (landed by the robosuite session; until it's on `main`, read the verified code in [`docs/plans/2026-10-22-sim-shared-core.md`](../plans/2026-10-22-sim-shared-core.md)):
  - `dataengine/variation.py`: `VariationSpec`, `EpisodeConfig`, `object_poses`, `train_configs`, `eval_configs`, `TIERS`;
  - `dataengine/expert.py`: `plan_waypoints`, `grasp_pose`, `Q_DOWN`;
  - `dataengine/geometry.py`: quaternions in wxyz.
  - The robosuite-specific parts (7-D delta actions, `LeRobotSink` Panda features) **don't apply** to this track.
- **Eval:** later, wrap the scene as `IsaacBackend(contracts.SimBackend)` with stage flags reach/grasp/lift/transport/place (Akanksh's harness).
- **Dataset contract:** LeRobot v3 + `manifest.yaml` + `meta/episodes_extra.jsonl`; never commit data to git.

## 4. Tips for agents

- **Tooling:** run headless Python scripts, not the GUI. Use the GUI only for looking around and checking prim paths.
- **Grippers:** the Robotiq drive joint is `right_gripper_ctrl`; the five mimic joints follow it. Never command the mimic joints.
- **Physics:**
  - set explicit masses: provided props have `mass = 0` and PhysX infers it from density;
  - give the syringe convex-hull colliders, and turn on CCD if it tunnels;
  - grasp the syringe across the barrel near its middle.
- **IK:**
  - use damped least squares (λ ≈ 0.05);
  - clamp the step to ≤ 2 cm and ≤ 0.1 rad per tick;
  - respect the joint limits above (joint 6 is 25°…265°).
- **Determinism:** seed everything from `cfg.seed`, and reset object velocities on every reset.
- **Frames:** the stage is **Y-up**. Every conversion to the shared code's z-up task frame happens in `sim/isaac/frames.py` only.

## 5. Done = report back to the master session

Push `feat/<you>-isaac-<topic>` branches with small PRs. Then post to the master session:
- per-object expert success;
- throughput;
- the pick-zone corners in world coordinates;
- the Isaac Sim version;
- any schema deviations.

**Rules:**
- Don't change `contracts/` (if truly needed, a separate PR via Rakesh).
- Tests run without Isaac (mark real-sim tests `@pytest.mark.gpu`).
- Commit as yourself; no AI `Co-authored-by` trailers (`git config core.hooksPath .githooks`).
