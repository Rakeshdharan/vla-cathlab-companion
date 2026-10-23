# CathLab Control Station, Part B: Multi-Engine Platform, Objects, Eval/Rollout Loop, Agent Gates (Design Spec)

*Approved 2026-10-23 ~10:20 (≈H20). Extends [the control station design](2026-10-23-control-station-design.md) (Plan A: orchestrator core, agent harness, UI shell). Owner: Rakesh.*

## 1. Goal

By the end of the hackathon, a user can do all of the following in the Control Station UI, with no coding:
- configure the task (engine, embodiment, objects, expert, variation, model);
- generate expert data in **robosuite, MuJoCo, Newton or Isaac**;
- convert it to LeRobot;
- train **SmolVLA / ACT**;
- evaluate the same checkpoint across engines (sim2sim);
- promote and roll back versions, and run live rollouts;
- drive rollouts from the **awareness assistant**;
- get a **failure report** that creates the next targeted run.

Anything the UI does not offer is one step away from an agent: **Build with agent** on a non-ready option, or the **Agent Brief** box. Both are backed by **stage gates**: contracts, conformance tests and reference implementations that any coding agent can follow to extend the platform (new engine such as Genesis or ManiSkill, new object source, new data method, new metric).

## 2. Decisions

| # | Decision |
|---|---|
| P1 | **Approach C:** our own thin engine-plugin layer now; RoboVerse/MetaSim can be added later as one more engine plugin, and its asset converter as an optional object exporter |
| P2 | **Engines fully runnable:** `robosuite` (exists), `mujoco` (standalone: MuJoCo Menagerie Panda + MJCF), `newton` (Newton 1.6 via the RoboSimStudio venv; robosuite's Panda exported for Newton), all built here. `isaac` (FR3 Duo, provided scene) built by Navneet against the same contract. Others (Genesis, ManiSkill3, MetaSim, PyBullet) through the `add_engine` harness template |
| P3 | **Engine manifest** `engines/<id>/engine.yaml` (env, worker, embodiments, io profile, capabilities stage → status, entry module, credits) and a **uniform CLI** `python -m engines.<id>.cli {preview\|check-expert\|generate\|eval\|rollout} --run <run.yaml> --out <dir> [--shard i --num-shards k]`. The UI and control plane are driven by manifests only |
| P4 | **I/O profiles:** `panda_libero` (7-D normalized end-effector delta, 8-D state, agentview + eye_in_hand 256², 20 Hz) for robosuite, MuJoCo and Newton; `fr3_duo_16d` (Siemens schema, 30 Hz) for Isaac |
| P5 | **Object package** `assets/objects/<name>/`: `object.yaml` + `visual.obj` + `collision/*.obj` + generated `object.xml` (MJCF). Importers: **primitive** (dimensions), **mesh** (OBJ/STL/GLB → trimesh normalise → CoACD convex hulls → MJCF), **Objaverse** (search by name → GLB → mesh importer). Engines load the MJCF: robosuite `MujocoXMLObject`, MuJoCo include, Newton `add_mjcf`, Isaac Lab `MjcfConverter`/`MeshConverter` on the Isaac worker |
| P6 | **Data-quality rules** in the shared sink/converter for every engine: drop no-op frames, require success to hold for 10 steps, validate counts when aggregating, record versions and the git commit in metadata |
| P7 | **Expert:** shared waypoint expert + per-engine `ExpertSim` adapter + `engines/<id>/expert.yaml` parameters; **Check expert** button (N configs → success per object) |
| P8 | **Training:** SmolVLA fine-tuned from `HuggingFaceVLA/smolvla_libero` (baseline) and ACT, using `lerobot-train` recipes on the GPU worker (token auth); writes a ModelCard; W&B tracking |
| P9 | **Eval:** policy server (lerobot env) ⇄ `PolicyClient` ⇄ engine `SimBackend`; suite v2 (3 objects × 5 tiers × 4 seeds per engine); the expert as reference; the same Panda checkpoint in robosuite, MuJoCo and Newton gives cross-physics sim2sim |
| P10 | **Registry:** versions `cathlab-pick@vN`; **Promote** rule (beats production beyond the 95 % CI and p95 latency ≤ budget); **Rollback**; **Rollout** = live episode with video |
| P11 | **ASSIST:** awareness `Assistant` (procedure step manual or from the VLM) proposes or accepts an object → instruction → rollout of the production model in the chosen engine |
| P12 | **Failure report:** failing cells (engine × tier × object × layout/colour), stage histogram, wrong-object grasps, sim2sim gap, VLM explanations, example videos, recommendations; **Create next run** with exactly those cells; exported as `report.md` + `report.json` |
| P13 | **Stage gates** `harness/gates/<stage>.yaml` for engine, scene, objects, expert, data_method, convert, intel, train, eval, rollout, assist, report: extension point, contract, conformance test, context files, reference implementation, credits |
| P14 | **Agent Brief** (deterministic): request + stage/engine ticks → markdown brief (gate contract, conformance test, matching templates, files, conventions, commands). Copy, Download .md, or Run with agent (`free_task`). Also on the CLI: `python -m harness brief` |
| P15 | **Repo-wide agent instructions:** an AGENTS.md "Extending" section, `docs/EXTENDING.md`, `.github/instructions/{engines,objects}.instructions.md`, and the repo skill `.github/skills/cathlab-extend/SKILL.md` |
| P16 | **Docker:** images for control, frontend, robosuite, mujoco, newton (CUDA), lerobot (CUDA); `compose.yaml`. The agent and Isaac stay on their hosts |

## 3. Components

| Path | Responsibility |
|---|---|
| `engines/registry.py` | Load and validate manifests (`EngineManifest`), capability queries, I/O profiles |
| `engines/cli_base.py` | Shared argparse and stage dispatch, so each engine implements only stage functions |
| `engines/robosuite/` | Manifest + `cli.py` wrapping `sim/robosuite` and the existing scripts |
| `engines/mujoco/` | Scene from Menagerie Panda + object packages, OSC-like delta controller (Jacobian IK), `ExpertSim`, `SimBackend`, cameras |
| `engines/newton/` | Scene via RoboSimStudio (Panda, table, tray, `add_mjcf` objects), delta-pose IK → joint targets, `ExpertSim`, `SimBackend`, cameras; runs in the RoboSimStudio venv |
| `engines/isaac/` | Manifest + CLI skeleton for Navneet (FR3 Duo, `fr3_duo_16d`) |
| `assets/objects/<name>/`, `assets/importers/{primitive,mesh,objaverse}.py` | Object packages and importers |
| `dataengine/npz_sink.py`, `scripts/npz_to_lerobot.py` | Extended with the P6 data-quality rules and I/O profiles |
| `dataset_intel/` | Validators, coverage over the variation grid, readiness score |
| `serve/policy_server.py`, `serve/policy_client.py`, `serve/registry.py` | Policy serving, client adapter, versions, promotion and rollback |
| `train/recipes/{smolvla,act}.yaml`, `train/run.py` | Training wrapper + ModelCard |
| `eval/suite_v2.yaml`, `eval/` | Suite v2 and runner glue, built on Akanksh's metrics and judge |
| `report/failure_report.py` | Failure report and next-run generator |
| `orchestrator/stages.py`, `orchestrator/pipeline.py` | Stage registry (capability, command builder, outputs) and dispatch to workers |
| `harness/gates/*.yaml`, `harness/brief.py`, `harness/tasks/{add_engine,add_object_importer,add_data_method,add_eval_metric,add_report_section}.yaml` | Gates, Agent Brief, new templates |
| `frontend/` | PIPELINE, DATA, MODELS, REPORT and ASSIST tabs wired; Agent Brief box on every tab; object import dialog |
| `docker/`, `compose.yaml` | Containers |

## 4. Error handling

- A stage is disabled with a reason when its engine lacks the capability, its worker is offline, or an input artifact is missing.
- Every job has a state, a log and its outputs. A failed stage blocks the stages that depend on it.
- Importers validate meshes (watertight warning, scale bounds, hull count limit) and refuse to overwrite existing packages.
- The policy server returns 503 until the model is loaded; the client retries for up to 60 s.

## 5. Testing

- **Core** (`uv run pytest -q`, GPU- and simulator-free): manifests, registry, CLI dispatch with a fake engine, importers (primitive; mesh with a tiny cube OBJ; Objaverse with a mocked index), sink rules, intel, registry and promotion, policy server/client with a dummy policy, failure report, brief assembly, gate files valid, every template's gate exists.
- **Engine conformance** (per engine env): `tests/engines/test_conformance.py --engine <id>` resets a config, steps, checks obs shapes/keys and stage flags, and requires the expert ≥ 80 % on 5 configs per object.
- **End-to-end smoke:** one run in robosuite of 6 episodes → convert → intel → a 50-step ACT train → 6-episode eval → promote → rollout video.

## 6. Credits (added to CREDITS.md)

- MuJoCo Menagerie (Franka Panda)
- Newton physics
- RoboSimStudio (Newton Panda + scene tooling)
- CoACD
- obj2mjcf (approach)
- trimesh
- Objaverse
- Isaac Lab MJCF/Mesh converters
- RoboVerse/MetaSim (approach C option)
- NVlabs MimicGen and GraspGen (harness methods)
- LeRobot SmolVLA `smolvla_libero` checkpoint
