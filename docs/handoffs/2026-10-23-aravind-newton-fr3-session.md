# Session Handover: Aravind: FR3 Duo in Newton → synthetic data at scale

> **Status 2026-10-23 17:30: done** ([dataset](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1), [report](https://github.com/aravinds1610/mujoco_cathlab_vla/blob/main/docs/NEWTON_REPORT.md)). **Continue with [part 2](2026-10-23-aravind-part2-mujoco-and-training.md):** FR3 MuJoCo task dataset + FR3 model training on a cloud GPU.

*From the master session, 2026-10-23 ~14:30 (≈H23). Owner: **Aravind**, in **his own repo** [`aravinds1610/mujoco_cathlab_vla`](https://github.com/aravinds1610/mujoco_cathlab_vla). The output is a dataset on HF. Paste this file into your agent as the first message.*

## 1. Why this, why now

The challenge robot is the **FR3 Duo**, and the team needs FR3 data for "Hand me the {bottle | vial | syringe}". Where things stand:

| Track | State |
|---|---|
| robosuite, **Panda** | ✅ 384 episodes ([`Rakeshdharan56/pick_robosuite@v1`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite)); 5 models fine-tuning on the RTX 6000 (SmolVLA, ACT on HF; X-VLA, π0.5, MolmoAct2 running or queued) |
| Isaac Sim, **FR3 Duo** | ❌ no working episodes yet; can't run on your machine |
| **Your MuJoCo FR3 Duo** | ✅ robot, scene, IK expert, 16-D Siemens schema, 200 episodes (v1) |

So **you are our FR3 data source.** The plan:
- move your FR3 scene into **Newton** (NVIDIA + Google DeepMind + Disney Research, Linux Foundation) using its **MuJoCo-Warp solver**;
- run **many randomised copies of the scene in parallel on the GPU**, with batched cameras;
- generate **more, more varied** FR3 episodes for our task.

**What Newton gives us that plain MuJoCo doesn't:**
1. **Throughput:** dozens of worlds per GPU step instead of one CPU episode at a time.
2. **A second physics for the same robot and scene file.** Running the same planned episodes in MuJoCo (CPU) and Newton (MuJoCo-Warp) gives an FR3 sim2sim number.
3. **An NVIDIA-stack story without Isaac's install pain.**

The Panda-on-Newton item (B8) is dropped from Rakesh's plan, so **you own Newton**.

**Time:** freeze is ≈ H32 (≈ 23:20 tonight). FR3 data must reach the GPU PC early enough to train tonight. **Deadlines are in §5. Hold the go/no-go at 16:30. The fallback (§5 N1) guarantees data either way.**

## 2. Verified facts (checked 2026-10-23 14:20 on the master PC, Newton 1.6.0)

| Fact | Detail |
|---|---|
| Install | `pip install "newton[sim]==1.6.0"`. `[sim]` brings `mujoco` 3.12 + `mujoco_warp` 3.12 and needs Python 3.12. PyPI also has 1.7.0, but **every API below was checked on 1.6.0**, so pin it |
| **Version clash with your repo** | Your `requirements.txt` pins `mujoco==3.3.7` and `lerobot==0.6.1`. **Make a separate venv for Newton** (`.venv-newton`, §4) and keep your current venv for LeRobot conversion. Same split as our robosuite pipeline: raw `.npz` → convert |
| MJCF import | `newton.ModelBuilder.add_mjcf(source, ...)`. `source` is a **file path or an XML string**, so `builder.add_mjcf(build_spec(...).to_xml())` works with your `MjSpec` scene. The importer handles **equality constraints and tendons** (the Robotiq 2F-85 needs both) and actuators |
| Your robot models | Newton's own test suite covers Menagerie **`franka_fr3_v2`** and **`robotiq_2f85`** (`newton/tests/test_menagerie_mujoco.py`, classes `TestMenagerie_FrankaFr3V2`, `TestMenagerie_Robotiq2f85`). Read them first: they show the import flags and any known quirks |
| Parallel worlds | Build one scene into a `scene = newton.ModelBuilder()`, then `main.replicate(scene, world_count, spacing=...)` (signature checked). Physics: `newton.solvers.SolverMuJoCo(model, separate_worlds=True, ...)` |
| Cameras | `newton.sensors.SensorTiledCamera(model)` renders every camera in every world in one batched call. Reference: `newton/examples/sensors/example_sensor_tiled_camera.py` (`utils.create_color_image_output(w, h, camera_count)`, per-camera FOV). **`add_mjcf` does not import `<camera>` tags**: set the 3 cameras up yourself from `cathlab_sim/isaac_reference.json` (head pose; wrist pose in the flange frame). You already compute these in `scene.py` |
| Contacts / sensors | `newton.sensors.SensorContact` exists, for the grasp flag |
| Robot-arm example | `newton/examples/robot/example_robot_panda_hydro.py` (Panda in Newton) |

## 3. Decisions (made with Rakesh; don't re-open)

| # | Decision |
|---|---|
| A1 | **Arm: RIGHT.** Same as the Isaac track and the Siemens demos. ⚠️ Your v1 expert uses the **left** arm (`ScriptedExpert(side="left")`, tray at (0.58, −0.26) placed for it). Switch to `side="right"`, mirror the pick region and tray for the right arm's reach, and keep the left arm at `HOME_Q` (still recorded). Probe reach first (N2) |
| A2 | **Task objects = the team's: bottle, vial, syringe.** All three are always present: one target, the other two as distractors.<br>• **bottle:** upright cylinder Ø77 × 152 mm, 0.30 kg (the provided scene's Boston round bottle);<br>• **vial:** Ø27 × 42 mm, 0.02 kg;<br>• **syringe:** lying composite, barrel Ø15 × 100 mm + plunger Ø8 × 40 mm + flange 30 × 10 × 3 mm, total length 140 mm, 0.015 kg.<br>**Your own instruments** (sheath, guidewire, scissors, gauze) may be **extra distractors** for clutter. **Never the catheter** (NaN instability) |
| A3 | **Schema = your v1 = the Siemens `fr3_duo_16d`** (the provided Siemens datasets, e.g. `Rakeshdharan56/fr3_duo_handover_84ep`):<br>• `action` and `observation.state` float32[16]: left j1–7, left gripper, right j1–7, right gripper. Joints in rad; **action = absolute joint targets**; gripper **1 = open, 0 = closed**;<br>• cameras `observation.images.{external_camera, left_wrist_camera, right_wrist_camera}` at 224×224 RGB;<br>• **30 fps**, `robot_type: fr3_duo`;<br>• per-frame `success`, `task` = instruction. Keep your **std = 1 for constant columns** fix |
| A4 | **Variation grid** (same idea as robosuite and Isaac, so results compare):<br>• **10 layouts** (seeded; 8 train, **8–9 held out**), **5 object-colour palettes** (4 train, **4 held out**), **5 table colours** (4 train, **4 held out**);<br>• per episode: mass × U(0.7, 1.3), friction U(0.5, 1.0), light intensity × U(0.7, 1.3), colour temperature U(4000, 7500) K;<br>• **train grid = 3 objects × 8 × 4 × 4 = 384** cells.<br>The reference implementation is our [`dataengine/variation.py`](https://github.com/Rakeshdharan/vla-cathlab-companion/blob/9deeb21b6cc38cddb7abc74b2a9d27ac73120b76/dataengine/variation.py) + [`configs/variation/cathlab_v1.yaml`](https://github.com/Rakeshdharan/vla-cathlab-companion/blob/9deeb21b6cc38cddb7abc74b2a9d27ac73120b76/configs/variation/cathlab_v1.yaml) (pure numpy/pydantic). Copy it into your repo with a one-line credit; it's team code. Write your own `configs/variation/cathlab_fr3_newton.yaml` with **your** table/tray/pick-region coordinates. Keep `reach_corridor_width: 0.15` (the bottle must not stand between the robot and the target) and use `generation_order()` so any first-N slice is object-balanced |
| A5 | **Instructions:** "Hand me the {object}", "Please pass the {object}", "Can you give me the {object}" (template index = `seed % 3`) |
| A6 | **Data-quality rules** (same as Isaac D8):<br>• drop no-op frames: joint-target change < 1e-4 rad on all joints **and** gripper command unchanged;<br>• success must **hold for 15 frames** (0.5 s);<br>• keep only successful episodes, but log every attempt;<br>• record versions (newton, mujoco_warp, git commit) per episode |
| A7 | **Output:** HF dataset **`aravindsuresh16/cathlab_fr3_newton_v1`** (LeRobot v3), public is fine. Tag it **`v1` and `v3.0`**: LeRobot won't load a Hub dataset without the codebase tag. `meta/episodes_extra.jsonl` holds the full episode config (object, layout_id, colour ids, mass/friction/light values, seed, tier, engine). `manifest.yaml` has `source: scripted`, `scene_id: cathlab_fr3_newton` |

## 4. Setup (once)

```bash
cd mujoco_cathlab_vla
python3.12 -m venv .venv-newton && source .venv-newton/bin/activate
pip install "newton[sim]==1.6.0" pydantic pyyaml pillow imageio pytest
python -c "import newton, warp, mujoco, mujoco_warp; print(newton.__version__, warp.__version__, mujoco.__version__); import warp as wp; wp.init(); print(wp.get_cuda_devices())"
python -c "import newton, os; print(os.path.dirname(newton.__file__))"   # open the examples/tests listed in §2
```

Your existing `.venv` (mujoco 3.3.7, lerobot 0.6.1) stays for `--convert` / upload. **Never install both into one env.** Put new Newton code in `cathlab_sim/newton/` with a lazy `import newton`, so your current tests don't need it.

## 5. Tasks (in order; each ends with a ✅ check; times are targets)

**N1: Newton port, go/no-go (by 16:30)**
- `cathlab_sim/newton/scene.py`: `build_newton(spec_xml, n_worlds)`:
  - `scene.add_mjcf(build_spec(scene_spec).to_xml())` into one scene builder;
  - `main.replicate(scene, n_worlds)`;
  - `SolverMuJoCo`, timestep 1/600, 20 substeps per 30 Hz frame (as in your MuJoCo scene);
  - `SensorTiledCamera` with external + both wrist cameras at 224².
- Hold `HOME_Q` with grippers open for 300 frames, then close/open the right gripper on an empty hand. Save one 3-camera image per world for 4 worlds.
- ✅ Objects rest (drift ≤ 2 mm), no NaN, the gripper opens and closes, and images match your MuJoCo renders.
- **No-go = any blocker you can't fix by 16:30** (e.g. the gripper tendon/equality chain not actuating, cameras unusable). In that case:
  - **run N2–N4 in your existing MuJoCo backend** instead (CPU, multiprocessing, one process per core) so the FR3 data still arrives tonight;
  - write down exactly what blocked Newton.

**N2: Right-arm task + expert (by 18:00)**
- Switch to the right arm and the team objects (A1, A2).
- Reach probe: right-arm IK to the 4 corners of the new pick region and to the tray centre, 5 cm above the table, gripper down. Fix the region and tray so every corner is reachable within 1 cm.
- **Planning stays in MuJoCo, execution goes to Newton.** Your expert is kinematic: Cartesian waypoints → IK per frame from ground-truth poses.
  1. Plan each world's 16-D joint-target trajectory with your existing `ScriptedExpert(side="right")` on a CPU `MjData` of the same model.
  2. Stream those targets into the matching Newton worlds.
  3. Judge success from Newton's state (stage flags as in your `stages.py`).
- Fix the v1 grasp misses: your diag showed fingers closing on air.
  - Grasp across the narrow axis.
  - The bottle (Ø77 mm, inside the 85 mm opening) needs a side grasp below its centre of mass, approaching from the robot side, or a top grasp on the neck if the side fails.
  - Add a check-and-regrasp after lift if time allows.
- ✅ Expert **≥ 80 % per object** on 20 train configs each. Write `reports/newton_expert_check.json` with per-object success, mean steps and failure stage counts.

**N3: Variation (by 18:00, parallel with N2)**
- `cathlab_sim/newton/randomize.py` applies an episode config per world:
  - poses from the layout;
  - object colours (palette) and table colour;
  - mass and friction;
  - light intensity and colour temperature (Kelvin → RGB);
  - optionally ±1 cm / ±2° head-camera jitter.
- If per-world colours aren't supported by the tiled renderer, **batch worlds by (palette, table colour)**: one launch per combination; the grid has only 16. Log it.
- ✅ An 8-config preview grid PNG with visibly different colours, lighting and layouts, plus a 3-camera image of one world.

**N4: Generate + publish (first slice by 19:00, full by 21:00)**
- Generate in `generation_order`. Write raw per-episode `.npz` (keys: `state`, `action`, `external_camera`, `left_wrist_camera`, `right_wrist_camera`, `success`, `task`) + `meta/episodes_extra.jsonl` + `manifest.yaml`. Then convert in your old venv to LeRobot v3 (A3, A7).
- **Volume:**
  1. **First slice: 96 episodes** (the first 96 of `generation_order`, all 3 objects), pushed by **19:00** and announced, so the GPU PC can start an FR3 SmolVLA run.
  2. **Then the full 384-cell train grid.**
  3. **Then "more episodes"** at 2 more seeds per cell (resampled mass/friction/light, ±1 cm pose jitter) → up to **1,152**, as throughput allows. Push as the same repo with tag `v1.1`, or as `..._v2`.
- ✅ `LeRobotDataset("aravindsuresh16/cathlab_fr3_newton_v1", revision="v1")` loads with `action (16,)`, `state (16,)`, 3 images `(3, 224, 224)`, fps 30, `robot_type fr3_duo`. Check `meta/info.json` shows `codebase_version: v3.0`. Include one preview GIF (external + right wrist).

**N5: The advantage numbers (by 22:00, short report)**
- **Throughput:** episodes/hour in Newton (GPU, n worlds) vs your MuJoCo (1 CPU process), on your laptop GPU.
- **FR3 sim2sim:** run the same planned trajectories for 60 configs in MuJoCo (CPU) and Newton (MuJoCo-Warp). Report success per object in each, plus the agreement rate.
- **Expert success:** train vs held-out tier (layouts 8–9 / palette 4 / table colour 4).
- ✅ `docs/NEWTON_REPORT.md` in your repo, plus a short message to Rakesh with the numbers and the HF link.

## 6. Out of scope

- Training: the RTX 6000 trains on your dataset. Tell Rakesh when the first slice is up.
- Isaac Sim.
- Code in Rakesh's repo: you don't need it; copy `variation.py` as in A4.
- GraspGen.
- Mesh objects: only if everything else is done.
- The catheter.

## 7. Rules

- Commit as yourself in your repo, with no AI `Co-authored-by` trailers.
- Datasets, videos and checkpoints never go into git.
- Keep a `CREDITS` line for Newton (Apache-2.0), MuJoCo / MuJoCo-Warp (Apache-2.0), Menagerie models (their own licences) and the copied `variation.py`.
- Log every non-obvious choice: arm/tray placement, grasp changes, world count, Newton workarounds.
- **Ask Rakesh** if a decision in §3 blocks you. Don't silently change the schema (A3): the training side depends on it.
