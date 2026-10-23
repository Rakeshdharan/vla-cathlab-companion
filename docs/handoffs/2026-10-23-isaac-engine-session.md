# Session Handover: Isaac Sim Engine (RTX 4090 PC): provided FR3 Duo scene → expert → data → eval → rollout

> **Status 2026-10-23 H23.5: DROPPED.** Isaac Sim produced no working run, and the RTX 4090 now hosts the ManiSkill3 + Genesis engines ([handoff](2026-10-23-rtx4090-maniskill-genesis-session.md)). FR3 Duo data comes from Aravind's MuJoCo/Newton track ([handoff](2026-10-23-aravind-newton-fr3-session.md)). Kept for reference.

*From the master session, 2026-10-23 ~10:30 (≈H20.5). **Supersedes** [`2026-10-22-isaac-fr3-session.md`](2026-10-22-isaac-fr3-session.md) and [`2026-10-22-navneet-gpu-machine.md`](2026-10-22-navneet-gpu-machine.md) (keep them for the verified USD facts). **Owner: Navneet** (in parallel with his awareness work); Rakesh helps. On the RTX 4090 PC, paste this file into the agent as the first message, or say "follow `docs/handoffs/2026-10-23-isaac-engine-session.md`".*

## 1. Goal

Make **Isaac Sim an engine plugin** of the CathLab Control Station, so the UI and orchestrator can run every stage on it exactly as on robosuite, MuJoCo and Newton:

> **preview → check-expert → generate (shards) → convert to LeRobot → eval → rollout**

The scene is the **provided Siemens FR3 Duo hospital scene**. The **right arm only** does "Hand me the {bottle | vial | syringe}": it picks the named object from a zone on the right supply cart and places it in `ServingTray`. Data is recorded in **the provided Siemens schema**, so it can be co-trained with the 94 provided demos.

The architecture is in [`docs/specs/2026-10-23-multi-engine-platform-design.md`](../specs/2026-10-23-multi-engine-platform-design.md) (Part B, decisions P2–P7, P9–P10, P13). The engine contract you implement is **fully specified in §4 below**. The shared `engines/` registry and UI wiring are built in parallel on the master PC (Plan B, task B1) and will read your manifest unchanged.

**Out of scope here:**
- training the FR3 model (separate training session on the RTX 6000);
- the robosuite, MuJoCo and Newton engines;
- the UI;
- `contracts/` changes;
- `dataengine/` changes (ask Rakesh; see §8).

## 2. Read first (in repo, on `main`)

| Doc / code | Why |
|---|---|
| [`AGENTS.md`](../../AGENTS.md) | Repo rules (lazy heavy imports, GPU-free default tests, no AI co-author trailers, data never in git) |
| [`2026-10-22-isaac-fr3-session.md`](2026-10-22-isaac-fr3-session.md) §1 | **Verified USD facts:** prim paths, bboxes, cameras, joint names and limits, gripper drive, ready pose |
| [`docs/data.md`](../data.md) | The provided dataset schema you must match |
| `sim/isaac/objects.py` + `tests/test_isaac_objects.py` | **Done (I1):** bottle, vial and syringe set up on the stage; box parked. Not yet verified against the real USD |
| `dataengine/variation.py` | `VariationSpec`, `EpisodeConfig`, `object_poses`, `train_configs`, `eval_configs`, `TIERS`, `generation_order` |
| `dataengine/expert.py` | The `ExpertSim` protocol, `ExpertParams`, `plan_waypoints`, `run_expert_episode(sim, spec, cfg, params, q_down, record=...)` |
| `dataengine/npz_sink.py`, `dataengine/generate.py` | Raw `.npz` episodes + `meta/episodes_extra.jsonl` + `manifest.yaml` (format reference) |
| `sim/robosuite/` + `sim/robosuite/README.md` | **The reference engine.** Copy its structure (scene, randomize, expert_sim, backend, params) |
| `contracts/sim.py`, `contracts/reports.py` | `SimBackend`, `StepInfo`, `EvalReport` |
| `eval/runner.py` | `run_suite(policy, sim, scene, task, seeds, model_version, dataset_versions)` for policy eval |

## 3. Decisions (don't re-open)

| # | Decision |
|---|---|
| D1 | Scene = provided `Hackathon_FR3Duo_flattened.usd`, unchanged except our objects. Get it from HF `Rakeshdharan56/cathlab_env_isaac` (private: ask Rakesh for access) or the team share. **Never commit USD.** The stage is **Y-up**, in metres |
| D2 | **Right arm only**; the left arm holds its ready pose `[0, −45, 0, −135, 0, 90, 45]°` and is still recorded |
| D3 | Objects: bottle = scene `NaturalBostonRoundBottle_A02_PR_NVD_01`, vial = scene `SM_PillBottle_01v`, syringe = primitive composite `/World/Cathlab/Syringe`. Already done in `sim/isaac/objects.py` |
| D4 | **I/O profile `fr3_duo_16d`** (= Siemens): `observation.state` and `action` are `float32[16]` in the order left j1–7, left gripper, right j1–7, right gripper. Joints in rad; **action = absolute joint targets**; gripper normalised **1 = open, 0 = closed** (`norm = 1 − q/0.8`, drive joint `right_gripper_ctrl`, 0…0.8 rad). Cameras `observation.images.{external_camera, left_wrist_camera, right_wrist_camera}` at **224×224** RGB. **30 fps** recording from **60 Hz** physics. `success: bool`, `robot_type: fr3_duo` |
| D5 | **Expert = the shared waypoint expert.** It emits normalised 7-D end-effector deltas, and your `IsaacExpertSim.step(a7)` turns them into right-arm joint targets with differential IK (damped least squares). **What gets recorded is the 16-D joint-target action**, not the 7-D delta (see §5 I4) |
| D6 | The **variation grid** has the same schema as robosuite (`configs/variation/cathlab_v1.yaml`) but its own geometry: `configs/variation/cathlab_fr3_isaac.yaml`. Train = 3 objects × 8 layouts × 4 object colours × 4 cart-top colours = **384**. Held out: layouts 8–9, palette 4, cart colour 4. Lighting ranges as in the provided data (§5 I3) |
| D7 | Generation order = `dataengine.variation.generation_order` (as robosuite), so `--start/--n` windows and shards are comparable across engines |
| D8 | **Data-quality rules** (Part B P6): drop no-op frames (joint-target change < 1e-4 rad on all joints **and** gripper command unchanged); success must hold for **15 frames** (0.5 s at 30 Hz); validate frame and video counts on conversion; record the Isaac Sim version, Isaac Lab version and git commit per episode |

## 4. The engine contract (what the orchestrator will call)

### 4.1 Manifest: `engines/isaac/engine.yaml`

```yaml
id: isaac
label: "Isaac Sim — provided FR3 Duo scene (right arm)"
env: {kind: command, python: "${ISAAC_PYTHON}"}      # e.g. ~/isaacsim/python.sh or Isaac Lab's ./isaaclab.sh -p
worker: isaac
embodiments: [fr3_duo]
io: fr3_duo_16d
variation: configs/variation/cathlab_fr3_isaac.yaml
task: configs/tasks/cathlab_pick.yaml
scene_asset_env: CATHLAB_ISAAC_USD                    # absolute path to Hackathon_FR3Duo_flattened.usd
capabilities:                                         # status per stage: ready | in_progress | planned
  preview: in_progress
  objects: {primitive: in_progress, mjcf: planned}
  randomize: planned
  expert: {analytic: planned, graspgen: planned}
  generate: planned
  eval: planned
  rollout: planned
entry: engines.isaac.cli
credits:
  - {name: "NVIDIA Isaac Sim", url: "https://developer.nvidia.com/isaac/sim"}
  - {name: "Isaac Lab", url: "https://github.com/isaac-sim/IsaacLab", license: "BSD-3-Clause"}
  - {name: "Siemens Healthineers hackathon FR3 Duo scene", url: "", license: "restricted, not redistributed"}
```

Flip a capability to `ready` only when its acceptance check (§5) passes on the 4090 PC, and log the evidence in `docs/decisions.md`.

### 4.2 Uniform CLI: `python -m engines.isaac.cli <stage> ...` (run with `${ISAAC_PYTHON}`)

Every stage runs **headless**. It exits 0 on success and non-zero with a one-line reason on stderr, prints progress lines to stdout (the worker streams them to the UI), and writes its outputs under `--out`. Optional `--run var/runs/<id>/run.yaml` (a Control Station run snapshot): when given, read `selection.objects` to restrict objects. Explicit flags win.

| Stage | Arguments | Outputs in `--out` |
|---|---|---|
| `preview` | `--spec <variation.yaml> --n 8` | `grid.png` (external camera after reset + settle, n train configs + 1 all-held-out), `preview.json` (configs used) |
| `check-expert` | `--spec --split train --n 20 --objects bottle,vial,syringe [--param k=v ...]` | `expert_check.json`, **same format as `scripts/robosuite_expert_check.py --json`**: `{"split", "params", "per_object": {obj: {"success", "attempted", "rate", "mean_steps", "max_steps"}}, "failures": {obj: {waypoint: count}}}` |
| `generate` | `--spec --split train --start 0 --n 384 --shard i --num-shards k --dataset-id cathlab/pick_isaac_fr3 --version v1 --created-by <name>` | `raw/shard_<i>/episodes/episode_*.npz`, `raw/shard_<i>/meta/episodes_extra.jsonl`, `raw/shard_<i>/manifest.yaml` (`contracts.DatasetManifest`, `source: scripted`) |
| `convert` *(run in the **lerobot** env, not Isaac's)* | `--dataset <out of generate> --repo-id <hf id> [--push]` | `lerobot/` (LeRobot v3, `fr3_duo_16d` features, fps 30), tags `v1` + `v3.0` when pushed |
| `eval` | `--spec --suite configs/eval/suite_v2.yaml --policy {expert\|random\|url:<policy-server-url>} --model-version <id> --out` | `eval_report.json` (`contracts.EvalReport`; `scene_variant` = tier), `videos/` (failed episodes, external camera, mp4) |
| `rollout` | `--policy ... --object vial --seed 5000 [--tier in_distribution]` | `rollout.mp4` (external + right wrist side by side), `rollout.json` (`{success, steps, failed_stage, latency_ms_p50/p95}`) |

Until the policy server exists (Plan B task B4; it will expose a `PolicyClient` in `serve/policy_client.py` implementing `contracts.PolicyAdapter`), `eval` and `rollout` must work with `--policy expert` (an adapter wrapping the scripted expert) and `--policy random`. Accept `url:` but fail with a clear "policy server not available yet" message if `serve.policy_client` cannot be imported.

## 5. Tasks (in order; each ends with a ✅ check)

**I0 — Machine setup** (do once, then log in `docs/decisions.md`)

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks
git config user.name "<your name>" && git config user.email "<your email>"
uv sync && uv run pytest -q                  # GPU-free core tests must pass
export ISAAC_PYTHON=~/isaacsim/python.sh     # or: <IsaacLab>/isaaclab.sh -p
export CATHLAB_ISAAC_USD=/abs/path/Hackathon_FR3Duo_flattened.usd
$ISAAC_PYTHON -m pip install pydantic pyyaml numpy imageio fastapi uvicorn httpx   # repo deps inside Isaac's python
```

- Record the Isaac Sim version (4.5 uses `omni.isaac.*`, 5.x uses `isaacsim.*`), the Isaac Lab version and the driver/CUDA version.
- Headless smoke: `SimulationApp({"headless": True})`, open the USD, step 100 frames, save one PNG per camera.
- ✅ 3 camera PNGs; versions logged.

**I1 — Verify objects against the real USD** (`sim/isaac/objects.py` exists)
- Call `setup_objects(stage)`, step 300 frames, and fix any wrong prim path or size.
- If the syringe tunnels or jitters: enable CCD, thicken the colliders slightly.
- ✅ All three objects rest on cart `SM_SupplyCart_01e` with drift ≤ 2 mm; before/after PNGs in the PR.

**I2 — Task frame, pick zone, variation spec**
- **Reach probe:** with right-arm diff-IK, drive `right_tcp` to the 4 corners of a candidate pick zone on cart 01e (left of the tray) and to the tray centre, 5 cm above the cart top, gripper pointing down. Measure the cart-top height (expected ≈ y 0.966).
- `sim/isaac/frames.py`: `task_to_world(pose7)` and `world_to_task(pose7)`. The **task frame** has its origin at the pick-zone centre on the cart surface; z = world +Y (up); x and y lie in the cart plane, with x pointing from the robot base towards the cart. Quaternions are **wxyz**.
- `configs/variation/cathlab_fr3_isaac.yaml`: copy `cathlab_v1.yaml` and set:
  - `variation_id: cathlab_fr3_isaac`;
  - `scene.*` in the task frame: tray centre and inner size from the USD (≈ 0.229 × 0.307 m), the pick region from the probe, `min_separation`, `reach_corridor_width`;
  - bottle `radius: 0.0385, length: 0.152`, vial `radius: 0.0135, length: 0.042`, syringe as in v1.
- ✅ Test `tests/sim/test_isaac_frames.py` (no Isaac needed): round-trip is exact; every `object_poses(spec, cfg)` for the 384 train configs lies inside the pick zone and outside the tray. Zone corners logged in world coordinates.

**I3 — Randomizer** `sim/isaac/randomize.py`: `apply_episode_config(stage, spec, cfg)` sets:
- object poses (task → world);
- object colours (palette `cfg.obj_color_id`, overriding the bound material's diffuse colour);
- cart-top colour (`cfg.table_color_id`);
- mass × `mass_scale`;
- friction on the shared physics material;
- lights: `RectLight` intensity 1,886–4,066, `DomeLight` 350–613, colour temperature 4,393–7,905 K, rect tilt ±13°, mapped from `cfg.light_intensity` / `cfg.light_temp_k`.

Use direct USD attribute edits (reliable on every reset), or `omni.replicator.core` randomizers if you prefer. ✅ `preview` writes a grid of 8 visibly different configs.

**I4 — Expert adapter** `sim/isaac/expert_sim.py`: `IsaacExpertSim` implements `dataengine.expert.ExpertSim`.
- `reset(cfg)`: reset the robot to the ready pose, apply the config, settle 30 physics steps, return obs.
- `get_object_pose(name)` / `get_ee_pose()`: `[x, y, z, qw, qx, qy, qz]` in the **task frame**. The end-effector pose is `right_tcp`.
- `step(a7)`:
  1. de-normalise (`dpos = a[:3]·max_dpos`, `drot = a[3:6]·max_drot`);
  2. target pose = current TCP + delta (rotation applied in the task frame);
  3. right-arm DLS IK (λ ≈ 0.05; clamp ≤ 2 cm and ≤ 0.1 rad per tick; respect joint limits, joint 6 is 25°…265°) → right joint targets;
  4. gripper `a[6] > 0` → `right_gripper_ctrl` target 0.8 rad, otherwise 0;
  5. the left arm keeps the ready pose;
  6. step physics 2 ticks (60 Hz → one 30 Hz frame);
  7. return obs + `StepInfo`.
- Keep `self.last_action16`, the 16-D joint-target command (gripper normalised), so the recorder stores it (D5).
- `tray_contains(name)`: object at rest (linear speed < 0.02 m/s, angular < 0.5 rad/s) inside the tray inner bounds.
- Stage flags:
  - reach: TCP within 5 cm of the grasp point;
  - grasp: both finger pads in contact with the target;
  - lift: target bottom > 5 cm above the cart;
  - transport: target xy inside the tray;
  - place: released and at rest in the tray.
- Obs = `{"state": float32[16], "external_camera", "left_wrist_camera", "right_wrist_camera": uint8[224,224,3]}`.
- `q_down`: measure `right_tcp`'s orientation when pointing straight down and pass it to `run_expert_episode(..., q_down=...)`. Never edit the shared expert.
- Parameters live in `engines/isaac/expert.yaml`. Start from `max_dpos: 0.02, max_drot: 0.2, max_steps: 450` (30 Hz) and tune tolerances and offsets there.
- ✅ `check-expert --n 20` → **≥ 80 % per object**, failure waypoints logged, the `expert: analytic` capability set to `ready`.

**I5 — Recorder + generate** `engines/isaac/recorder.py` + the CLI `generate`:
- `run_expert_episode(..., record=cb)` calls `cb(obs, a7)` before each step. In your callback, **store the frame after the step using `sim.last_action16`** (Siemens convention: state at frame k, action = joint target applied for frame k).
- Write the `.npz` per episode with keys `state`, `action`, `external_camera`, `left_wrist_camera`, `right_wrist_camera`, `success` (bool per frame), plus `meta/episodes_extra.jsonl` and `manifest.yaml` in the `dataengine/npz_sink.py` layout. Use a module-level key map in `engines/isaac/recorder.py`; don't change `NpzSink`.
- Apply D8: no-op drop, success hold, version info.
- ✅ `generate --n 6 --num-shards 1` → 6 episodes, one GIF preview.

**I6 — Convert** `engines/isaac/convert.py` (lerobot env):
- Read the raw shards → `LeRobotDataset.create(repo_id, fps=30, robot_type="fr3_duo", features=FR3_FEATURES)`. The features must equal the provided dataset's `meta/info.json` features (names included), **except** `raw_state_row` / `raw_action_row`.
- Validate counts.
- With `--push`: upload with tags `v1` and `v3.0`.
- ✅ The 6 episodes load with `LeRobotDataset`, and their feature dict matches `Rakeshdharan56/fr3_duo_handover_84ep`'s (same keys, shapes, names).

**I7 — Eval backend + rollout** `sim/isaac/backend.py`: `IsaacBackend(contracts.SimBackend)`:
- `reset(scene, seed, target_object)` selects the config with `eval_configs(spec, tier, [seed])`;
- `step(action16)` applies the joint targets directly (policies output 16-D);
- `render()` returns the external camera.
- CLI `eval` wraps `eval.runner.run_suite` per tier; `rollout` records an mp4.
- ✅ `eval --policy expert` on 3 objects × 5 tiers × 2 seeds writes a valid `EvalReport`; `rollout --policy expert` writes an mp4.

**I8 — Full dataset**
- Generate the 384 train configs in shards (as many processes as fit in 24 GB of VRAM), convert, and push `pick_isaac_fr3` with tags `v1` and `v3.0` (the HF org comes from Rakesh).
- Log expert success per object, throughput (episodes/min/GPU) and the HF link in `docs/decisions.md`.
- ✅ Dataset on HF; capabilities `preview`, `randomize`, `expert.analytic`, `generate`, `eval` and `rollout` set to `ready`.

**I9 — Worker** (`orchestrator/worker.py` with token auth is on `main`). On the 4090 PC:

```bash
export CATHLAB_WORKER_TOKEN=<shared secret from Rakesh>
$ISAAC_PYTHON -m orchestrator.worker --name isaac --capabilities preview,check-expert,generate,eval,rollout \
    --host 0.0.0.0 --port 8713 --token "$CATHLAB_WORKER_TOKEN"
```

The master PC points `configs/workers.yaml` (`isaac`) at `http://<4090-pc>:8713`, or uses an SSH tunnel (`ssh -N -L 8713:localhost:8713 <4090-pc>`). ✅ The UI header shows the ISAAC lamp green.

**I10 — Object packages** (after Plan B task B2 lands `assets/objects/<name>/object.xml`): `objects.mjcf` capability.
- Import any object package with Isaac Lab's `MjcfConverter` (or `MeshConverter` on `visual.obj` + `collision/*.obj`) into `/World/Cathlab/<name>`, with mass and friction from `object.yaml`.
- ✅ A new package (e.g. a guidewire hoop) appears in `preview` and the expert reaches ≥ 80 % on it.

**Stretch (after I8):**
- I11: Replicator/Cosmos Transfer visual augmentation on ~50 episodes;
- I12: Isaac Lab Mimic re-targeting;
- I13: GraspGen grasp ranking as `expert.graspgen`.

## 6. Tests

- GPU-free tests (`uv run pytest -q`) must stay green. Put Isaac-free logic (frames, key maps, no-op filter, manifest writing, CLI argument parsing) in plain-Python modules with tests in `tests/sim/test_isaac_*.py` and `tests/engines/test_isaac_*.py`.
- Tests that need Isaac go under `@pytest.mark.gpu` and run with `$ISAAC_PYTHON -m pytest -m gpu tests/sim -k isaac`.

## 7. Report back to the master session (after I4, I6 and I8)

Small PRs on `feat/<you>-isaac-<topic>` branches. Post:
- the capability statuses;
- expert success per object;
- throughput;
- pick-zone corners (world);
- Isaac Sim / Isaac Lab versions;
- any schema deviation;
- the HF dataset link.

## 8. Rules

- Don't change `contracts/` or `dataengine/`. If the shared expert or sink truly needs a change, open an issue or ask Rakesh; the same code drives robosuite, MuJoCo and Newton.
- Don't touch `sim/robosuite/`, `engines/{robosuite,mujoco,newton}/` or `configs/variation/cathlab_v1.yaml` (frozen; the published v1 dataset depends on it).
- Import Isaac lazily, inside functions, so the repo imports without it.
- Never commit the USD, datasets, videos or `.env`. Commit as yourself; the hook strips AI `Co-authored-by` trailers.
- The provided Siemens material is restricted: HF repos built from it stay **private**.
