# Mitesh — Synthetic Data Engine, Environment Factory, New Objects

## Agent brief (paste to your agent)

> You are helping build the CathLab Companion learning loop (spec: `docs/specs/2026-10-22-cathlab-learning-loop-design.md`, §5.2 and §8.1).
>
> My scope:
> - (1) `dataengine/`: generators that output LeRobot datasets + `DatasetManifest` for the FR3 Duo handover task, in priority order:
>   - domain randomization replaying the provided controller (poses, clutter, lighting, distractors, camera jitter);
>   - MimicGen / Isaac Lab Mimic (DexMimicGen for bimanual) re-targeting of the provided demos;
>   - a scripted expert using GraspGen grasps + cuRobo motion (this is also our "own expert policy");
>   - teleop and Cosmos Transfer as documented optional generators.
> - (2) `envfactory/`:
>   - prompt → `SceneSpec` → USD/MJCF scene (agentic, e.g. RoboSimStudio agent / Isaac Lab tooling);
>   - a manual scene builder from YAML;
>   - new object creation: library lookup, else image/text-to-3D, then fix scale/collision/mass and register the asset.
>
> Every generator must accept a `GapSpec` (object, region, clutter, stage) so the loop can request targeted data.

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| M1 | Generator interface `dataengine.generate(gap: GapSpec \| None, n: int, method: str) -> DatasetManifest`, plus a registry of methods with status in the option catalogue | `dataengine/__init__.py` | H4 |
| M2 | Domain randomization generator (first and fastest) → `cathlab/handover_dr@v1` | `dataengine/domain_rand.py` | H12 |
| M3 | Scripted expert (GraspGen + cuRobo or IK fallback) as both a generator and a `PolicyAdapter` (`ScriptedExpert`) for eval | `dataengine/scripted_expert.py` | H16 |
| M4 | MimicGen/Isaac Lab Mimic re-targeting from the provided demos | `dataengine/mimicgen.py` | H20 |
| M5 | Env factory: `envfactory.from_prompt("cathlab side table with 8 instruments, one tray") -> SceneSpec` + generated scene; `envfactory.from_yaml(path)` | `envfactory/` | H22 |
| M6 | New object pipeline: `envfactory.objects.create(name, prompt_or_image) -> asset_path` with physics props; add ≥1 new cathlab object (e.g. a hemostatic valve) | `envfactory/objects.py`, `assets/objects/` | H24 |
| M7 | Option catalogue table (all engines incl. teleop + Cosmos, chosen vs documented, cost/speed/robustness) | `docs/research/data_engines.md` | H28 |

## Interfaces

- **Consumes:**
  - `GapSpec` (from Sharanrajapur's coverage map / Rakesh's loop);
  - `SceneSpec`, `TaskSpec`, `DatasetManifest`;
  - `IsaacBackend` (Rakesh) to run generation;
  - `configs/robots/fr3_duo.yaml`.
- **Produces:** datasets with `source` ∈ {`domain_rand`, `mimicgen`, `scripted`, `teleop`, `cosmos`}; `ScriptedExpert(PolicyAdapter)`; scenes under `configs/scenes/`.

## Acceptance

- ≥1,000 synthetic episodes pass the 02 readiness gate by H20.
- Given a `GapSpec` from the loop, the engine produces K targeted episodes without manual steps.
- One prompt yields a new scene where the eval runner runs `ScriptedExpert` successfully, which is the "any task / any env" demo.
