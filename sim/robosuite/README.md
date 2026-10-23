# sim/robosuite

robosuite track for the cathlab pick task (Franka Panda, LIBERO I/O). Owner: Rakesh.

| Module | What |
|---|---|
| `env.py` | `CathlabPickPlace(ManipulationEnv)`: table, static tray, bottle/vial/syringe built from `VariationSpec`; raised home pose; `agentview` + `robot0_eye_in_hand` 256², 20 Hz, `OSC_POSE` |
| `randomize.py` | `apply_episode_config(env, spec, cfg)`: poses, object/table colours, mass, friction, lighting |
| `expert_sim.py` | `RobosuiteExpertSim`: the `dataengine.expert.ExpertSim` adapter (core frame, wxyz, stage flags) |
| `backend.py` | `RobosuiteBackend`: `contracts.SimBackend` for policy eval; `reset(scene, seed, obj)` → `eval_configs(spec, tier, [seed])` |
| `params.py` | `ROBOSUITE_EXPERT_PARAMS` tuned for OSC |

Conventions:
- Core frame = world − (−0.56, 0, 0.8): robot base frame with the table top at z = 0.
- Expert EE = grip site; state = TCP position + hand axis-angle + finger joints.
- Images use the opencv convention (upright).

Requires the `robosuite` conda env (Python 3.10, robosuite 1.5.2, **mujoco 3.3.7**) and `MUJOCO_GL=egl`. Tests in `tests/sim/` skip without robosuite.

Data collection: [docs/tasks/robosuite-collection.md](../../docs/tasks/robosuite-collection.md).
