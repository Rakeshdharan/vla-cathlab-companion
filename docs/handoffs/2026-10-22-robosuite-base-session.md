# Session Handover: robosuite Base (Scene → Expert → Variation → First Episodes)

*From the master planning session, 2026-10-22 ~21:20 (≈H7.5). Owner: Rakesh. Paste this file into the new session as the first message, or say "follow docs/handoffs/2026-10-22-robosuite-base-session.md".*

## Goal of this session

Build the **robosuite base end-to-end** for our task, so newer teammates can then scale variations and episode collection without needing robotics knowledge:

1. our own scene with the 3 finalized objects;
2. the scripted expert controller doing the task reliably;
3. per-episode variation (synthetic-data randomization);
4. a few recorded base episodes in LeRobot v3 format;
5. a short "how to collect more" guide for teammates.

**Out of scope here** (stays in the master session): the Isaac track (Mitesh), Akanksh's eval-harness review, Navneet's plan feedback, the HF organization, training.

## Read first (in repo)

| Doc | Why |
|---|---|
| [`docs/specs/2026-10-22-sim-framework-design.md`](../specs/2026-10-22-sim-framework-design.md) | **The design.** Task, I/O contract, scene dimensions, variation grid, expert, milestones |
| [`docs/plans/2026-10-22-sim-shared-core.md`](../plans/2026-10-22-sim-shared-core.md) | **Verified code** for the shared core (variation, expert, fake sim, generator, LeRobot sink). Copy it verbatim |
| [`AGENTS.md`](../../AGENTS.md) | Repo rules (contracts frozen, GPU-free tests, no AI co-author trailers) |
| [`docs/decisions.md`](../decisions.md) | Why we chose this design |

## Decisions already made (do not re-open)

- **Robot:** single-arm Franka Panda + Franka hand. **Task:** "Hand me the {bottle | vial | syringe}". Pick the named object from the table and place it in the tray; all 3 objects are always present.
- **I/O (LIBERO-compatible):**
  - action `float32[7]` normalized [-1, 1]: `[dx,dy,dz]/0.05 m`, `[drx,dry,drz]/0.5 rad` axis-angle in the base frame, gripper −1 open / +1 close;
  - state `float32[8]` = EEF position, EEF axis-angle, 2 finger joints;
  - cameras `agentview` + `eye_in_hand` at 256×256;
  - 20 Hz, max 300 steps.
- **Objects** (from `configs/variation/cathlab_v1.yaml`):

  | Object | Geometry | Mass |
  |---|---|---|
  | bottle | upright cylinder Ø62 × 250 mm | 0.30 kg |
  | vial | upright cylinder Ø27 × 42 mm | 0.02 kg |
  | syringe | lying flat; barrel Ø15 × 100 mm + plunger Ø8 × 40 mm + flange 30 × 10 × 3 mm, total length 140 mm | 0.015 kg |

- **Scene:**
  - table 0.8 × 1.2 m;
  - tray inner 0.25 × 0.32 m, 0.04 m walls, centred at base-frame `(0.55, -0.30)`;
  - pick region 0.30 × 0.40 m centred at `(0.55, 0.15)`.
- **Variation:** train grid 3 objects × 8 layouts × 4 object colours × 4 table colours = **384**. Held out: layouts 8–9, object colour 4, table colour 4. Mass ×U(0.7, 1.3), friction U(0.5, 1.0), light intensity ×U(0.7, 1.3), colour temperature U(4000, 7500) K.
- **Expert:** privileged poses + per-class grasp templates (bottle side grasp, vial top grasp, syringe top grasp across the barrel) → 8 waypoints → normalized deltas. GraspGen comes later as "expert v2".

## Verified environment facts (checked 21:20)

| Fact | Value |
|---|---|
| Conda env | `robosuite`, Python 3.10.22 |
| Versions | robosuite **1.5.2**, MuJoCo **3.3.7** |
| Offscreen render | Works with `MUJOCO_GL=egl` (256×256 uint8 for `agentview` + `robot0_eye_in_hand`) |
| Default Panda controller | `load_composite_controller_config(controller="BASIC", robot="Panda")` → arm `OSC_POSE`, `input_type: delta`, `input_ref_frame: base`, `output_max: [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]`. **Exactly our action convention; pass the expert's action unchanged** |
| Action dim | 7 (6 arm + 1 gripper; gripper −1 open / +1 close) |
| Missing in the env | `pydantic`, `pyyaml`, `imageio` → `pip install pydantic pyyaml imageio` |
| LeRobot | **Not installed** in the `robosuite` env. It's installed in the `lerobot` env (0.5.2, editable from `~/lerobot`). See B5 for the recording strategy |
| Harmless warnings | missing private macros file, `robosuite_models`, mink IK |

## robosuite gotchas (read before coding)

1. **Quaternion order:** robosuite observations (`robot0_eef_quat`, `*_quat`) are **xyzw**. MuJoCo `data.xquat` / `body_xquat` are **wxyz**. Our core (`dataengine/geometry.py`) is **wxyz** everywhere, so convert at the adapter boundary.
2. **Frames:** robosuite observations are in the **world** frame, but our core expects the **robot base frame with the table top at z = 0**. Define one transform in the adapter: `p_core = p_world − [base_x, base_y, table_top_z]`.
   - Panda base rotation is identity in robosuite, so deltas need no rotation.
   - Get `base_x`, `base_y` from the robot base body position after reset.
   - `table_top_z` = the table offset z.
3. **TCP:** `robot0_eef_pos` is the grip site between the fingertips, i.e. our TCP. Check the EEF orientation after reset. If the resting "gripper down" quaternion differs from `Q_DOWN = [0, 1, 0, 0]` (wxyz) by a yaw, pass the measured one as `q_down` to `run_expert_episode` / `generate_scripted`; don't edit the core.
4. **Images are upside down** by default (OpenGL convention). Set `robosuite.macros.IMAGE_CONVENTION = "opencv"` before `suite.make`, or flip `[::-1]`. Log the choice in `docs/decisions.md`; we'll match LIBERO preprocessing at fine-tune time.
5. **Placing objects:**
   - don't use robosuite's random placement sampler;
   - after `env.reset()`, set each object's free-joint qpos from `dataengine.variation.object_poses(spec, cfg)` (converted to world frame, wxyz), call `sim.forward()`, then step a few zero actions to settle.
6. **Model edits** (colour/mass/friction/light) go on `env.sim.model` (`geom_rgba`, `body_mass`, `geom_friction[:, 0]`, `light_diffuse`). Re-apply them on every reset. Use `hard_reset=False` for speed.
7. **Grasp check:** robosuite's `env._check_grasp(gripper=env.robots[0].gripper, object_geoms=obj)` is a ready-made contact-based "grasp" stage flag.

## Tasks (do in order; each ends with a check)

**B0 — Branch + unblock contracts + shared core**
- `git switch -c feat/rakesh-robosuite-base origin/main`.
- `git cherry-pick aad2849`. This is Akanksh's contracts commit: it keeps his authorship and its contents were verified identical to the foundation plan. It also brings `pyproject.toml` / `uv.lock`.
- Implement shared-core plan Tasks 1–5, copying the code from the plan doc.
- ✅ `uv run pytest -q` is green.
- ✅ The plan's Task 6 smoke run prints `384 / 384`.

**B1 — Scene** (`sim/robosuite/env.py`)
- `CathlabPickPlace(ManipulationEnv)` with:
  - the table;
  - the tray (static body with 4 walls + floor);
  - bottle and vial as `CylinderObject`;
  - the syringe as a composite (barrel + plunger + flange), lying flat;
  - cameras `agentview` (keep robosuite's default pose) + `robot0_eye_in_hand`, 256²;
  - `control_freq=20`, `horizon=300`, `ignore_done=True`.
- Dimensions come from `VariationSpec`, never hard-coded.
- ✅ A reset → 50 zero-action steps → saved `agentview` PNG shows all objects resting, with no penetration or explosion.
- **Share the `agentview` camera pose** (position + quaternion) in `docs/decisions.md` so Mitesh can match it in Isaac.

**B2 — Expert adapter** (`sim/robosuite/expert_sim.py`, `sim/robosuite/backend.py`)
- `RobosuiteExpertSim` implements `dataengine.expert.ExpertSim`: `reset(cfg)`, `get_object_pose`, `get_ee_pose`, `step`, `tray_contains`, with the frame and quaternion conversions above.
- `obs = {"state": (8,), "agentview": img, "eye_in_hand": img}`.
- `StepInfo` stage flags:
  - reach: TCP within 5 cm of the grasp point;
  - grasp: `_check_grasp`;
  - lift: object bottom > 5 cm above the table;
  - transport: object xy inside the tray;
  - place: released and at rest in the tray.
- `RobosuiteBackend(SimBackend)` wraps the same thing for policy eval.
- ✅ A test with `pytest.importorskip("robosuite")` runs one episode on `cfg = train_configs(spec)[0]` and checks obs shapes and frames (object z ≈ `center_z()` after settle).

**B3 — Expert works**
- Run `run_expert_episode` on 20 train configs per object.
- ✅ **≥80% success per object.**
- If it fails: tune `ExpertParams` (tolerances, offsets, close steps) and object friction first. Change the core grasp templates only if needed, and keep the fake-sim tests green.
- Record the per-object success and the failing waypoint (`ExpertResult.trace`) in `docs/decisions.md`.

**B4 — Variation** (`sim/robosuite/randomize.py`)
- `apply_episode_config(env, spec, cfg)` applies:
  - poses;
  - object colours (palette `cfg.obj_color_id`) and table colour;
  - mass × `mass_scale` and `friction`;
  - light intensity and colour temperature (simple Kelvin→RGB).
- ✅ A PNG grid of 8 different configs, visibly varied.
- ✅ The expert still reaches ≥80% on 20 mixed configs.

**B5 — Record a few base episodes**
- Script `scripts/robosuite_generate.py --split train --start 0 --n 20 --out data/pick_robosuite_base`, plus `--shard i --num-shards k` for later parallel runs.
- **Recording strategy:**
  - try `pip install -e ~/lerobot` in the `robosuite` env;
  - if it conflicts (numpy/mujoco/torch), add `NpzSink` (raw per-episode `.npz` + the same `episodes_extra.jsonl`), then a converter script run in the `lerobot` env that replays the npz files through `LeRobotSink`.
  - Record which path you used in decisions.
- ✅ 10–20 episodes load with `LeRobotDataset(..., root=...)`, with shapes `action (7,)`, `state (8,)`, images `(3, 256, 256)`, fps 20.
- ✅ One preview mp4 or GIF.
- Data never goes into git.

**B6 — Teammate collection guide** (`docs/tasks/robosuite-collection.md`)
- Written for teammates new to VLA/robotics, as copy-paste commands:
  1. env setup;
  2. run shard i of k;
  3. check the success rate and preview;
  4. merge shards;
  5. upload to HF as `cathlab/pick_robosuite@v1`.
- Include how to read a failure and whom to ask.
- ✅ A teammate can follow it without reading code.

## Done = report back to the master session

Push `feat/rakesh-robosuite-base` and open a PR. Then post a short summary back to the master session:
- per-object expert success;
- episodes/minute (single process);
- the `agentview` camera pose;
- the image-convention and recording-path decisions;
- anything that needed a core or contract change.

## Rules

- Don't change `contracts/` (if truly needed, a separate contracts PR).
- Don't touch `sim/isaac/`.
- Tests must run without a GPU or robosuite (`importorskip`).
- Commit as yourself; no AI `Co-authored-by` trailers (hook: `git config core.hooksPath .githooks`).
- Heavy data, videos and checkpoints stay out of git.
