# sim/maniskill

ManiSkill3 track for the cathlab pick task (Franka Panda, LIBERO I/O, **PhysX 5**). Owner: Rakesh. Engine plugin: [engines/maniskill](../../engines/maniskill/engine.yaml).

| Module | What |
|---|---|
| `env.py` | `CathlabPickEnv(BaseEnv)` (`CathlabPick-v1`) + `panda_cathlab` agent: table, static tray, bottle/vial/syringe from `VariationSpec`; `HOME_QPOS`; `agentview` + `eye_in_hand` 256², 20 Hz, `pd_ee_delta_pose` scaled to ±0.05 m / ±0.5 rad |
| `randomize.py` | `apply_episode_config(env, spec, cfg)`: object/table colours, mass (+inertia), friction, light intensity × Kelvin, poses |
| `expert_sim.py` | `ManiSkillExpertSim`: the `dataengine.expert.ExpertSim` adapter (core frame, wxyz, stage flags) |
| `backend.py` | `ManiSkillBackend`: `contracts.SimBackend`; `reset(scene, seed, obj)` → `eval_configs(spec, tier, [seed])` |
| `params.py` | `MANISKILL_EXPERT_PARAMS` |

Conventions (measured against robosuite; history on the pre-dev branch):
- World frame = core frame (base at z 0.112, table top at z 0).
- TCP = `panda_hand` ⊕ 0.097 m along hand z, orientation hand · Rz(−90°) (robosuite grip site). State = TCP position + hand axis-angle + `[q1, −q2]`.
- Action boundary: rotation **and** gripper signs flipped before `env.step` (ManiSkill 3.0.1 scales rotation by `rot_lower` < 0; gripper +1 = open).
- CPU PhysX, one env per process; parallelise by shards.

Install (Python 3.10; conda env `maniskill` or a venv):

```bash
pip install "mani_skill==3.0.1" "torch==2.9.1" pydantic pyyaml imageio imageio-ffmpeg pillow pytest   # torch cu128 for driver 570
vulkaninfo --summary   # must list the NVIDIA GPU (SAPIEN renders through Vulkan)
PYTHONPATH=. python -m pytest -q tests/engines/test_conformance.py -k maniskill tests/sim/test_maniskill_expert_sim.py
```
