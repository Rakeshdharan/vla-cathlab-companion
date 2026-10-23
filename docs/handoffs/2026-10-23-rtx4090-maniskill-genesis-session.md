# Session Handover: RTX 4090: ManiSkill3 (Mitesh) + Genesis (Akanksh) engines

*From the master session, 2026-10-23 ~14:50 (≈H23.5). **Isaac Sim is dropped** (no working run after a day; it replaces nothing on the critical path). This machine now hosts **two new Panda engines** built exactly like robosuite. Paste this file into each agent as the first message: "follow `docs/handoffs/2026-10-23-rtx4090-maniskill-genesis-session.md`, I own <ManiSkill3 | Genesis>".*

## 1. Why these two engines

All our other engines run **MuJoCo physics**: robosuite (Panda, done), Aravind's MuJoCo FR3 and Aravind's Newton (MuJoCo-Warp). Isaac was our only *different* physics, and it's gone. So the claim "a policy trained in one simulator works in another" (spec D5) currently has nothing to stand on.

| Engine | Physics | Why it's worth it | Owner |
|---|---|---|---|
| **ManiSkill3** 3.0.1 (SAPIEN 3.0.3) | **NVIDIA PhysX 5** (GPU or CPU) | The physics Isaac would have given us, without Isaac. GPU-parallel sim + render. Ships a Panda with an **end-effector delta-pose controller**, almost our action space | **Mitesh** |
| **Genesis** 1.4.5 | **Genesis' own GPU rigid solver** (`quadrants` backend) | A third, independent physics. Batched GPU rendering (Madrona `BatchRenderer`). Ships the Menagerie Panda MJCF | **Akanksh** |

**Both use the Panda and our `panda_libero` I/O.** So:
- the **5 checkpoints being trained on robosuite data** (SmolVLA, ACT, X-VLA, π0.5, MolmoAct2) can be **evaluated unchanged** in both engines. That cross-physics eval is the headline number;
- **the same 384-cell grid** generates paired data, and v2 = robosuite + ManiSkill + Genesis.

**Time:** it's ≈ H23.5; freeze is ≈ H32 (≈ 23:20). The deadlines in §6 are hard. If an engine misses its 17:00 go/no-go, its owner stops and joins the other engine.

## 2. Read first (in repo, on `main`)

| Doc / code | Why |
|---|---|
| [`AGENTS.md`](../../AGENTS.md) | Repo rules: lazy heavy imports, GPU-free core tests, no AI co-author trailers, data never in git |
| [`docs/handoffs/2026-10-22-robosuite-base-session.md`](2026-10-22-robosuite-base-session.md) | **The recipe you are repeating** (scene → expert → variation → episodes). Read its gotchas |
| `sim/robosuite/` (`env.py`, `randomize.py`, `expert_sim.py`, `backend.py`, `params.py`, `README.md`) | **The reference engine.** Mirror its structure as `sim/maniskill/` and `sim/genesis/` |
| `engines/registry.py`, `engines/cli_base.py`, `engines/robosuite/{engine.yaml,cli.py,evaluate.py}` | **The engine plugin contract** (manifest + uniform CLI: `preview`, `check-expert`, `generate`, `convert`, `eval`, `rollout`). Copy `engines/robosuite/` as your template |
| `tests/engines/test_conformance.py` | Your engine must pass it in your env (it picks up any manifest with a `backend:`) |
| `dataengine/{variation,expert,geometry,generate,npz_sink}.py` | Shared variation grid, shared expert (`ExpertSim` protocol, `run_expert_episode`), npz sink. **Don't edit them**; ask Rakesh |
| `configs/variation/cathlab_v1.yaml` (**frozen**), `configs/eval/suite_v2.yaml`, `configs/tasks/cathlab_pick.yaml` | Grid, eval suite (3 objects × 5 tiers × seeds 5000–5003), instructions |
| `scripts/npz_to_lerobot.py` | Engine-agnostic converter for `panda_libero` shards. Reuse it for `convert` |

## 3. The scene and I/O to reproduce (measured in robosuite, 2026-10-23 14:45)

Everything is in the **core frame**: the robot base frame, x forward, y left, z up, origin at the base xy, **z = 0 at the table top**. Quaternions are **wxyz**. Reproduce these numbers as closely as each engine allows; every deviation goes in `docs/decisions.md`.

| Item | Value |
|---|---|
| Panda base | at core (0, 0, **0.112**), identity rotation, i.e. the base sits 0.112 m above the table-top plane |
| Home joints | `HOME_QPOS = (0.0, -0.2917, 0.0, -2.5521, 0.0, 2.3877, 0.7854)` → TCP at core (0.45, 0, 0.40), gripper pointing down. Fingers open |
| TCP | **0.097 m from the `panda_hand` body origin along hand +z** (robosuite grip site). ManiSkill's `panda_hand_tcp` and Genesis' frames may differ by a few mm: compute the TCP yourself as hand pose ⊕ (0, 0, 0.097) |
| Table | top 0.8 × 1.2 m, centre core (0.56, 0, top at 0), 5 cm thick |
| Tray | static, inner 0.25 × 0.32 m centred at core (0.55, −0.30); floor 0.01 thick; walls 0.01 thick × 0.04 high |
| Objects | from `VariationSpec` (`cathlab_v1.yaml`), **never hard-coded**:<br>• bottle: upright cylinder r 0.031, h 0.250, 0.30 kg;<br>• vial: r 0.0135, h 0.042, 0.02 kg;<br>• syringe: lying composite, barrel r 0.0075 + plunger r 0.004 × 0.040 + flange 0.003 × 0.030 × 0.010, total length 0.140, 0.015 kg (see `sim/robosuite/env.py` `SyringeObject`).<br>Initial poses from `dataengine.variation.object_poses(spec, cfg)` (object centre, core frame) |
| `agentview` camera | position core **(1.06, 0, 0.55)**, looks along **(−0.7081, 0, −0.7061)**, up **(−0.7061, 0, 0.7081)** (look-at point (0.352, 0, −0.156)), **fovy 45°**, 256×256 |
| `eye_in_hand` camera | on `panda_hand`: position (0.05, 0, 0) in the hand frame, looks along hand **+z**, up hand **+x**, **fovy 75°**, 256×256 |
| Images | RGB uint8 **upright** (row 0 = top). SAPIEN and Genesis each have their own camera-axis convention: **build the cameras from position + look-at + up above, not from robosuite's quaternion** |
| Control | **20 Hz**; max 300 steps per episode (eval: 400) |
| **Action** `float32[7]` | `[dx, dy, dz] / 0.05 m`, `[drx, dry, drz] / 0.5 rad` (axis-angle, **base frame**, applied about the TCP, relative to the **current** TCP pose), gripper **−1 open / +1 close** |
| **State** `float32[8]` | TCP position (core frame), TCP axis-angle (from the hand quaternion, wxyz → axis-angle), 2 finger joints **as robosuite signs them**: `[+q_finger1, −q_finger2]` (open ≈ [0.04, −0.04]). ManiSkill and Genesis report both fingers positive, so **negate the second** |
| Obs dict (`ExpertSim`/`SimBackend`) | `{"state": (8,) float32, "agentview": (256,256,3) uint8, "eye_in_hand": (256,256,3) uint8}` |
| Stage flags | reach: TCP within 5 cm of the grasp point · grasp: both fingerpads touching the target · lift: target bottom > 5 cm above the table · transport: target xy in the tray · place: released and at rest in the tray. **Latched**, exactly as `sim/robosuite/expert_sim.py` |
| Expert | the **shared** `dataengine.expert.run_expert_episode`. The robosuite tuning lives in `sim/robosuite/params.py` (`release_height=0.05`, `min_grasp_tcp_z=0.013`). Start from it; your tuning goes in `sim/<engine>/params.py` |

## 4. Engine specifics (verified on 2026-10-23 from the installed wheels)

### 4.1 ManiSkill3 (Mitesh)

| Fact | Detail |
|---|---|
| Install | `pip install "mani_skill==3.0.1"` → `sapien` 3.0.3, `pytorch_kinematics` 0.7.6, `mplib`, `gymnasium`. Python ≥ 3.9. Apache-2.0 |
| **Vulkan is required** | SAPIEN renders through Vulkan. On WSL here it fails with `vk::createInstanceUnique: ErrorIncompatibleDriver`, and even scene building fails (`failed to find a rendering device`, because materials need the renderer). **First command on the 4090:** `vulkaninfo --summary` must list the RTX 4090. If it doesn't: `sudo apt install libvulkan1 vulkan-tools` and the NVIDIA driver's Vulkan ICD (`/usr/share/vulkan/icd.d/nvidia_icd.json`) |
| Panda | `robot_uids="panda"`, URDF `panda_v2.urdf`, EE link `panda_hand_tcp`. Control mode **`pd_ee_delta_pose`** (IK via `pytorch_kinematics` → joint PD targets; default `frame="root_translation:root_aligned_body_rotation"` = translation in the base frame, rotation base-aligned about the EE = **our convention**). With `use_target=False`, deltas are taken from the current EE pose, as in robosuite |
| **Action scaling differs** | Defaults are `pos_lower/upper = ±0.1 m`, `rot_lower/upper = ±0.1 rad` (normalised [−1, 1]). **Override to ±0.05 m and ±0.5 rad** in your agent/controller config (subclass the Panda agent or patch `_controller_configs`), so our action passes through unchanged |
| **Gripper sign is flipped** | `PDJointPosMimicControllerConfig(lower=-0.01, upper=0.04)`, normalised: **−1 = closed, +1 = open**. Ours is the opposite: send `-a[6]` |
| Rates | `SimConfig` default `sim_freq=100`, **`control_freq=20`** (= ours) |
| Scene building | Subclass `BaseEnv` (see `mani_skill/envs/tasks/tabletop/pick_cube.py`), register with `@register_env("CathlabPick-v1")`, build in `_load_scene`. Builders in `mani_skill.utils.building.actors.common`: `build_box`, `build_cylinder`; use `scene.create_actor_builder()` for the static tray (5 boxes) and the syringe composite. Cameras via `_default_sensor_configs` → `CameraConfig(uid, pose, width=256, height=256, fov=...)`; build `pose` with `sapien_utils.look_at(eye, target, up)` |
| Parallel | `sim_backend="gpu", num_envs=N` for many worlds. **Do correctness first with `sim_backend="cpu", num_envs=1`**, then parallelise generation by processes (shards, like robosuite). GPU-batched expert is a stretch |
| Physics knobs | Per-actor mass/friction via the builder's physical material and `set_mass` (CPU) / batched setters (GPU). Lighting via `_load_lighting` |

### 4.2 Genesis (Akanksh)

| Fact | Detail |
|---|---|
| Install | `pip install "genesis-world==1.4.5"` + a CUDA build of `torch`. Python 3.10–3.13, Apache-2.0. Pulls `quadrants==1.3.3` (Genesis' compute backend), `mujoco>=3.2.5` (MJCF parsing), `gs-madrona==0.0.10` (batch renderer, Linux x86_64). Use **its own env**: it must not share an env with robosuite (`mujoco==3.3.7`) |
| Init | `import genesis as gs; gs.init(backend=gs.gpu)` once per process |
| Panda | Built in: `gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml")` (also `panda_no_tendon.xml`, `hand.xml`). Joint names follow Menagerie (`joint1..7`, `finger_joint1/2`) |
| **No EE controller ships**: write one | Per step: target TCP = current TCP ⊕ (`a[:3]·0.05`, `exp(a[3:6]·0.5)` in the base frame) → `robot.inverse_kinematics(link=hand, pos=..., quat=...)` (target for the **hand** link, so subtract the 0.097 m TCP offset) → `robot.control_dofs_position(q[:7], arm_dofs)`. Gripper: finger target 0.04 if `a[6] < 0`, else close with `control_dofs_force` (≈ −20 N) or position 0. Set PD gains with `set_dofs_kp` / `set_dofs_kv` (start from Genesis' Franka examples). Clamp per-step joint change |
| Rates | `gs.options.SimOptions(dt=0.01)` → **5 sim steps per 20 Hz action** (`substeps` if contacts need it) |
| Cameras | `scene.add_camera(res=(256,256), pos=..., lookat=..., up=..., fov=...)`. Render with `cam.render(rgb=True)`. For many envs, `renderer=gs.renderers.BatchRenderer(use_rasterizer=True)` (Madrona). The wrist camera must follow the hand: `cam.attach(hand_link, offset_T)` (4×4 offset from §3) or re-pose it each step with `cam.set_pose(pos=..., lookat=..., up=...)` |
| Scene | `gs.morphs.Box` for the table and the 5 tray boxes (`fixed=True`), `gs.morphs.Cylinder` for bottle/vial. Syringe: 3 primitives welded, or a small MJCF/URDF string with 3 geoms. `scene.build(n_envs=N)` for parallel worlds (start with `n_envs=0`) |
| Physics knobs | Surface friction via `gs.materials.Rigid(friction=...)` / `gs.surfaces`; mass via the morph or `set_mass`; colours via `surface=gs.surfaces.Default(color=...)` |

## 5. Decisions (don't re-open)

| # | Decision |
|---|---|
| E1 | **Engine plugins**, exactly like robosuite: `engines/maniskill/{engine.yaml,cli.py,evaluate.py}` + `sim/maniskill/{env,randomize,expert_sim,backend,params}.py`; same with `genesis`. `io: panda_libero`, `variation: configs/variation/cathlab_v1.yaml`, `task: configs/tasks/cathlab_pick.yaml`, `env: {kind: conda, name: maniskill}` / `{name: genesis}`, `worker: <engine id>`, `backend: "sim.<engine>.backend:<Engine>Backend"`, capabilities honest (`planned` → `in_progress` → `ready` only after the check passes) |
| E2 | **Same grid and same episodes:** `train_configs(spec)` in `generation_order`, so robosuite / ManiSkill / Genesis episode *i* share the same config (paired). **Never edit `cathlab_v1.yaml`** |
| E3 | **Data:** raw shards via `dataengine.npz_sink.NpzSink` + `generate_scripted(..., source_sim="<engine>")`, converted with `scripts/npz_to_lerobot.py` in the `lerobot` env. Datasets **`Rakeshdharan56/pick_maniskill`** and **`Rakeshdharan56/pick_genesis`**, tags `v1` + `v3.0` (ask Rakesh for a write token, or push to your own account and tell him). Data never in git |
| E4 | **Data-quality rules** (Part B P6): keep only successful episodes; success must hold 10 steps; record engine/physics versions + git commit in `episodes_extra.jsonl` |
| E5 | **Eval:** `engines.<id>.cli eval --suite configs/eval/suite_v2.yaml --policy {expert,random,url:<server>}` → `contracts.EvalReport` with `scene_variant` = tier, same as `engines/robosuite/evaluate.py`. Policies through `engines.policies.make_policy` and `engines.expert_policy.ExpertPolicy` |
| E6 | **Isolation:** each engine in its own conda env (`maniskill`, `genesis`; Python 3.10 or 3.11 so `sim/` and `engines/` stay 3.10-compatible). `pip install pydantic pyyaml numpy imageio` there; run with `PYTHONPATH=<repo>`. `uv run pytest -q` (core) must stay green and engine-free: import the simulator **lazily** |

## 6. Tasks (each engine owner runs the same list; times are hard targets)

**E0: Machine + env (by 15:30)**
- Clone, `git config core.hooksPath .githooks`, set your git name/email, `uv sync && uv run pytest -q`.
- Create your conda env and install (§4). Mitesh: `vulkaninfo --summary` first.
- Smoke: ManiSkill `gym.make("PickCube-v1", robot_uids="panda", control_mode="pd_ee_delta_pose", obs_mode="rgb")`, 10 steps, save the camera image. Genesis: a Panda + plane scene, 100 steps, one camera image.
- Log in `docs/decisions.md`: GPU, driver, CUDA, engine version, Vulkan (ManiSkill).
- ✅ One rendered PNG per engine.

**E1: Scene (go/no-go by 17:00)**
- `sim/<engine>/env.py`: table, tray, three objects from `VariationSpec`, Panda at the §3 base pose and `HOME_QPOS`, both cameras per §3.
- Reset → place objects at `object_poses(spec, train_configs(spec)[0])` → settle 50 steps → save `agentview` + `eye_in_hand` PNGs.
- ✅ Objects rest (drift ≤ 2 mm, no penetration). The `agentview` PNG looks like robosuite's (same framing: tray left, objects centre-right). **Compare side by side with a robosuite render and put both in the PR.**
- **No-go at 17:00** = the scene can't be built or rendered → stop, document the blocker, join the other engine.

**E2: Adapter + expert (by 19:00)**
- `sim/<engine>/expert_sim.py`: `<Engine>ExpertSim` implements `dataengine.expert.ExpertSim`:
  - `reset(cfg)`, `get_object_pose`, `get_ee_pose` (TCP, core frame, wxyz), `step(a7)`, `tray_contains`;
  - stage flags per §3;
  - the action/state conversions of §3–4: scaling, gripper sign, finger sign, TCP offset.
- `sim/<engine>/backend.py`: `<Engine>Backend(SimBackend)` like `RobosuiteBackend`, so `reset(scene, seed, target)` runs `eval_configs(spec, tier, [seed])`.
- `check-expert --n 20` with `sim/<engine>/params.py`.
- ✅ **≥ 80 % per object** on 20 train configs. `expert_check.json` in the robosuite format. Failure waypoints logged in `docs/decisions.md`. `tests/engines/test_conformance.py -k <engine>` passes in your env.

**E3: Variation + preview (by 19:30, parallel with E2)**
- `sim/<engine>/randomize.py`: `apply_episode_config(env, spec, cfg)` applies:
  - object colours (palette `cfg.obj_color_id`) and table colour;
  - mass × `mass_scale` and friction;
  - light intensity × Kelvin colour;
  - re-applied every reset from nominal values (never compounding).
- ✅ `preview` writes `grid.png` with 8 visibly different configs + 1 all-held-out.

**E4: Generate + publish (first 96 by 20:30, all 384 by 22:00)**
- `generate` with `--shard i --num-shards k` (one process per shard; the 4090 has headroom for several).
- Then `convert` in the `lerobot` env, push with tags `v1` + `v3.0`, and load it back.
- ✅ `LeRobotDataset("Rakeshdharan56/pick_<engine>", revision="v1")` shows `action (7,)`, `state (8,)`, images `(3,256,256)`, fps 20, `robot_type panda`. Plus a preview GIF. Report episodes/min.

**E5: Cross-physics eval (by 23:00): the headline**
- `eval --policy expert` on `suite_v2` (reference row).
- Then the trained checkpoints, through the policy server once it exists (`--policy url:...`, Sharan's `serve/`), or the in-process `train/lerobot_adapter.py` if it has landed. Start with **SmolVLA and ACT** (already on HF: `Rakeshdharan56/smolvla_panda_pick_v1`, `Rakeshdharan56/act_panda_pick_v1`).
- ✅ `eval_report.json` per model with success by tier × object. One table: the same checkpoint in robosuite vs ManiSkill vs Genesis = the sim2sim gap.

## 7. Done = report back to the master session

Open one PR per engine from `feat/<you>-<engine>-engine`, then post:
- versions;
- the go/no-go result;
- expert success per object;
- episodes/min;
- the dataset link;
- the eval table (expert + checkpoints) per tier;
- every deviation from §3 (camera framing, TCP offset, controller differences).

## 8. Rules

- Stay in `engines/<engine>/`, `sim/<engine>/`, `tests/engines/`, `tests/sim/test_<engine>_*.py`, `docs/`.
- Don't edit `contracts/`, `dataengine/`, `configs/variation/cathlab_v1.yaml`, `sim/robosuite/`, `engines/robosuite/`: ask Rakesh.
- Simulator imports stay lazy. The core `uv run pytest -q` must not need ManiSkill or Genesis. Engine tests use `pytest.importorskip("mani_skill")` / `("genesis")`.
- Commit as yourself, with no AI `Co-authored-by` trailers. Conventional Commits. Rebase on `main` before pushing; `main` moves fast.
- Add `CREDITS.md` rows: ManiSkill3 / SAPIEN (Apache-2.0), Genesis (Apache-2.0), MuJoCo Menagerie Panda (Apache-2.0).
- Log every non-obvious choice as one line in `docs/decisions.md`.
