# Newton engine plugin (FR3 Duo) — design

*2026-10-23. Approved by the repo owner. Built through the agent harness (`add_engine`), reviewed by a human.*

## Goal

Add `engines/newton/`: Aravind's Newton 1.6 (MuJoCo-Warp, GPU) FR3 Duo cathlab scene as a regular engine plugin, so the
Control Station can preview, check the expert, generate, evaluate and roll out FR3 Duo policies in Newton like it does
for robosuite and ManiSkill. Source: [`aravinds1610/mujoco_cathlab_vla`](https://github.com/aravinds1610/mujoco_cathlab_vla)
at commit `fc019f87febda013aaa431f638b2a454aca36bb5` (handoff: [../handoffs/2026-10-23-aravind-newton-fr3-session.md](../handoffs/2026-10-23-aravind-newton-fr3-session.md)).

## Decisions

| # | Decision |
|---|---|
| D1 | Embodiment **FR3 Duo, right arm**; io profile **`fr3_duo_16d`** (16-D absolute joint targets, gripper 1 = open, 30 Hz, 3 cameras 224²). Not Panda. |
| D2 | **Vendor** Aravind's code (team code, his consent given) into `sim/newton/fr3/` at the pinned commit, imports re-pathed `cathlab_sim.*` → `sim.newton.fr3.*`. No runtime dependency on his repo. |
| D3 | Delivered by the harness: `python -m harness run add_engine --param engine_id=newton ...`. The agent builds; a human reviews the diff and pushes. |
| D4 | The shared Panda EEF expert (`dataengine.expert`) cannot drive the FR3. The engine uses Aravind's `TaskExpert` (IK plan on a CPU `MjData`, joint targets streamed into Newton). Deviation from the engine gate, recorded in `docs/decisions.md`. |
| D5 | `convert` uses the vendored FR3 converter (Siemens 16-D schema), not `scripts/npz_to_lerobot.py` (Panda-shaped). |
| D6 | Capabilities are honest: `ready` only with evidence from this machine (RTX 6000 Ada); otherwise `in_progress`. |

## Layout (all inside the `add_engine` template's allowed paths)

```
engines/newton/
  __init__.py
  engine.yaml        id newton, io fr3_duo_16d, embodiments [fr3_duo], env {kind: venv, name: ~/venvs/newton},
                     worker newton, variation/task -> sim/newton/configs/*, backend sim.newton.backend:NewtonFR3Backend
  cli.py             six stages via engines.cli_base.run_cli; no simulator import at module top level
  stages.py          preview / check-expert / generate / convert wrappers over sim.newton.fr3.newton.runner etc.
  evaluate.py        eval (suite_v2 tiers x objects x seeds -> contracts.EvalReport) and rollout
sim/newton/
  README.md          origin + pinned SHA, frame conventions, how to run
  __init__.py
  backend.py         NewtonFR3Backend(contracts.SimBackend), one world
  fr3/               vendored package (cathlab_sim/{scene,robot_io,layout,ik,expert,stages,spec_io,contracts}.py
                     + cathlab_sim/newton/*.py)
  configs/           cathlab_fr3_newton.yaml (variation), handover scene + objects, task.yaml (contracts.TaskSpec)
tests/engines/test_newton_cli.py   core-env tests: manifest validates, --help and every stage --dry-run work
```

## Components

- **Backend** `NewtonFR3Backend`: `reset(scene, seed, target_object)` builds the episode config for `seed` from the FR3
  variation and resets a 1-world `NewtonTaskSim`; `step(action[16])` → obs `{state (16,), external_camera,
  left_wrist_camera, right_wrist_camera (224,224,3) uint8}` and `StepInfo(stage_flags over contracts.STAGES, success)`;
  `render()` → external camera HxWx3; `close()` frees the Newton model.
- **Robot assets:** MuJoCo Menagerie from `$CATHLAB_MENAGERIE`, else `~/.cache/mujoco_menagerie` (the harness agent env
  doesn't forward `CATHLAB_*`). Missing → one-line error with the `git clone` command.
- **Stage outputs** follow `engines/README.md`: `grid.png` + `preview.json`; `expert_check.json`
  (`{split, params, per_object: {obj: {success, attempted, rate, mean_steps, max_steps}}, failures}`); raw npz shards +
  `meta/episodes_extra.jsonl` + `manifest.yaml` (`source: scripted`, `scene_id: cathlab_fr3_newton`); `lerobot/`;
  `eval_report.json`; `rollout.mp4` + `rollout.json`.
- **Policies:** `--policy expert | random | url:<server> | lerobot:<ckpt>` via `engines.policies.make_policy(spec, 16)`;
  `expert` is the vendored `TaskExpert` wrapped as a `PolicyAdapter`.

## Environment

`~/venvs/newton` (outside the repo, like `~/venvs/genesis`): Python 3.12, `newton[sim]==1.6.0`, `pydantic pyyaml numpy pillow
imageio[ffmpeg] pytest`. Commands run from the repo root so `contracts`, `engines`, `sim` import from the checkout.

## Verification

1. Core: `uv run pytest -q` green (no Newton/MuJoCo import reachable from core tests).
2. Conformance: `MUJOCO_GL=egl ~/venvs/newton/bin/python -m pytest -q tests/engines/test_conformance.py -k newton`.
3. Smoke: `preview --n 8`, `check-expert --n 5`, `rollout --policy expert` in `~/venvs/newton`; outputs as above.
4. Evidence row in `docs/decisions.md`; `CREDITS.md` rows: Newton (Apache-2.0), MuJoCo-Warp (Apache-2.0), Menagerie
   FR3 v2 + Robotiq 2F-85 (their licences), Aravind's `mujoco_cathlab_vla` @ `fc019f8`.
5. Catalog: `simulator.newton` and `embodiment.fr3_duo` status from the evidence.

## Out of scope / follow-ups (outside the template's allowed paths)

- `configs/workers.yaml`: a `newton` worker (`~/venvs/newton/bin/python -m orchestrator.worker --name newton ... --port 8716`).
- `engines/registry.py`: `obs_shapes` for `fr3_duo_16d` so conformance also checks image/state shapes.
- Panda on Newton; batched multi-world eval; MuJoCo-CPU FR3 engine (`mujoco` catalog item).
