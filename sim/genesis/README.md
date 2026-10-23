# sim/genesis

Genesis track for the cathlab pick task (Franka Panda, LIBERO I/O), mirroring `sim/robosuite`. Owner: Navneet.

| Module | What |
|---|---|
| `env.py` | `CathlabGenesisEnv`: table, static tray, bottle/vial/syringe from `VariationSpec`; Menagerie `panda_no_tendon.xml` at core (0, 0, 0.112), `HOME_QPOS`; `agentview` + `eye_in_hand` 256²; delta-EE controller (damped-LS IK at the TCP → joint PD), 20 Hz over 5 ms physics steps |
| `randomize.py` | `apply_episode_config(env, spec, cfg)`: poses, object/table colours, mass, friction, lighting, always from nominal values |
| `expert_sim.py` | `GenesisExpertSim`: the `dataengine.expert.ExpertSim` adapter (core frame, wxyz, latched stage flags) |
| `backend.py` | `GenesisBackend`: `contracts.SimBackend` for policy eval; `reset(scene, seed, obj)` → `eval_configs(spec, tier, [seed])` |
| `params.py` | `GENESIS_EXPERT_PARAMS` (`release_height` 0.065, `min_grasp_tcp_z` 0.013) |

Conventions (as robosuite):
- Core frame = world − (0, 0, 0.8): robot base frame with the table top at z = 0.
- Expert EE = grip-site frame: hand pose ⊕ (0, 0, 0.097), rotated −90° about hand z (fingers close along local x).
- State = TCP position + hand axis-angle + fingers `[+q1, −q2]`. Images upright RGB uint8.
- Action = `[dpos/0.05 m, drot/0.5 rad (base frame, about the TCP), gripper −1 open / +1 close]`.

Deviations from robosuite (controller dynamics, physics step, lighting, randomisation limits) are in
[docs/decisions.md](../../docs/decisions.md) (H25.5 rows).

Requires `~/venvs/genesis` (Python 3.11, genesis-world 1.4.5, CUDA torch); run with `PYTHONPATH=<repo>`. Engine
stages: `python -m engines.genesis.cli <stage>`. Tests in `tests/sim/test_genesis_env.py` skip without genesis.
**One Genesis process per GPU at a time** (the RTX 4090 dropped off the PCIe bus twice under parallel simulator load).
