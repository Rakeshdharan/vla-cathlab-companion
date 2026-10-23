---
description: "Scaffold a new SimBackend (MuJoCo, Newton, Genesis, ...); prefer the engine plugin flow in docs/EXTENDING.md with stage flags and a contract test."
argument-hint: "backend name, e.g. mujoco, newton, genesis"
agent: "agent"
---
Add a `SimBackend` for `${input:backend}`.

- Package: `sim/${input:backend}/`, with class `<Name>Backend(contracts.SimBackend)` implementing `reset(scene, seed, target_object)`, `step(action) -> (obs, StepInfo)`, `render()` and `close()`.
- `StepInfo.stage_flags` must compute `reach, grasp, lift, transport, place` from sim state, using the predicates in `configs/tasks/handover.yaml`. `success` is all stages true.
- Observation keys match `configs/robots/fr3_duo.yaml`.
- Import the simulator lazily. Add a `SceneSpec` example under `configs/scenes/` with `backend: ${input:backend}`; extend the `Backend` literal in `contracts/specs.py` via a contracts PR if needed.
- Tests: a contract-conformance test reusable for all backends (`tests/sim/test_backend_contract.py`, parametrised over available backends, with `DummySim` always included); real-sim tests marked `@pytest.mark.gpu`.
