# sim/newton

Newton track for the cathlab handover task (**FR3 Duo**, right arm, io `fr3_duo_16d`, **Newton 1.6 / MuJoCo-Warp**, GPU).
Engine plugin: [engines/newton](../../engines/newton/engine.yaml). The FR3 scene, IK expert and Newton runner are
Aravind's, vendored from [`aravinds1610/mujoco_cathlab_vla@fc019f8`](https://github.com/aravinds1610/mujoco_cathlab_vla/tree/fc019f87febda013aaa431f638b2a454aca36bb5)
(handoff: [docs/handoffs/2026-10-23-aravind-newton-fr3-session.md](../../docs/handoffs/2026-10-23-aravind-newton-fr3-session.md)).

| Module | What |
|---|---|
| `backend.py` | `NewtonFR3Backend`: `contracts.SimBackend`, one Newton world; `reset(scene, seed, obj)` → `eval_configs(spec, tier, [seed])` |
| `expert_policy.py` | `FR3ExpertPolicy`: the scripted expert as a `PolicyAdapter` (plans on a CPU MuJoCo twin, replays in Newton) |
| `fr3/` | Vendored `cathlab_sim` (imports `cathlab_sim.*` → `sim.newton.fr3.*`; each file starts with its origin line) |
| `fr3/newton/scene.py` | `NewtonTaskSim`: N parallel worlds, `SolverMuJoCo`, `SensorTiledCamera` (3 cameras, 224²) |
| `fr3/newton/cpu_sim.py`, `expert_fr3.py` | CPU MuJoCo twin + right-arm `TaskExpert` (IK per 30 Hz frame from ground-truth poses) |
| `fr3/newton/runner.py` | Plan on a CPU process pool, execute in Newton batches; `drop_noops`, `save_npz` |
| `fr3/newton/convert.py` | Raw npz (runner or engine shard layout) → LeRobot v3, Siemens `fr3_duo_16d` schema |
| `configs/variation/cathlab_fr3_newton.yaml` | FR3 variation grid: 10 layouts, 5 palettes, 5 table colours, held-out splits (same tiers as `suite_v2`) |

Conventions:
- World frame = FR3 Duo scene frame (Z up, robot faces +x, robot's left = +y; table top z = 0.8414 m), not the Panda core frame.
- `action` / `state` float32[16] = left j1–7, left gripper, right j1–7, right gripper; joints in rad, absolute targets;
  gripper 1 = open, 0 = closed. 30 Hz; physics 1/600 s, 20 substeps per frame.
- Cameras: `external_camera`, `left_wrist_camera`, `right_wrist_camera`, 224×224×3 uint8.
- Success = all stages latched in order and `place` held 15 frames (`fr3/newton/task_stages.py`).
- Changes vs upstream: Menagerie lookup, no writes into the repo, `versions()` records `vendored_from`, `convert.py`
  reads shard layouts and carries `features()` / `floor_constant_stats()`; v1-only modules (MuJoCo backend, generate,
  tools) are not vendored.

Install (Python 3.12, outside the repo):

```bash
python3.12 -m venv ~/venvs/newton
~/venvs/newton/bin/pip install "newton[sim]==1.6.0" pydantic pyyaml numpy pillow "imageio[ffmpeg]" pytest trimesh scipy
# CUDA 12.x driver: PyPI's warp-lang 1.18 needs a CUDA 13 driver, so use the cu12 build from the Warp release
~/venvs/newton/bin/pip install --force-reinstall --no-deps \
  https://github.com/NVIDIA/warp/releases/download/v1.18.0/warp_lang-1.18.0+cu12-py3-none-manylinux_2_28_x86_64.whl
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie ~/.cache/mujoco_menagerie   # or set $CATHLAB_MENAGERIE
PYTHONPATH=. MUJOCO_GL=egl ~/venvs/newton/bin/python -m pytest -q tests/engines/test_conformance.py -k newton
```

Run (from the repo root, `PYTHONPATH=. MUJOCO_GL=egl ~/venvs/newton/bin/python -m engines.newton.cli ...`):
`preview --out DIR`, `check-expert --out DIR --n 20 --worlds 16`, `generate --out DIR --n 96`, `eval --out DIR --policy expert`,
`rollout --out DIR --object vial`. `convert` needs a LeRobot env with `mujoco` (Aravind used lerobot 0.6.1 + mujoco 3.3.7).
The first Newton run JIT-compiles Warp kernels (~2 min, cached in `~/.cache/warp`).
