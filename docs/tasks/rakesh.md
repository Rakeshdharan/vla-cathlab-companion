# Rakesh — Lead: Architecture, Ingest, Training, Loop, Sim2Sim

## Agent brief (paste to your agent)

> You are helping build the CathLab Companion learning loop (spec: `docs/specs/2026-10-22-cathlab-learning-loop-design.md`).
>
> My scope:
> - (1) implement the foundation plan `docs/plans/2026-10-22-cathlab-loop-implementation.md` Tasks 1–4;
> - (2) `IsaacBackend(SimBackend)` for the provided FR3 Duo USD scene;
> - (3) `ingest/`: load the provided LeRobot dataset, validate it against `configs/robots/fr3_duo.yaml`, add paraphrased instructions and stage segmentation, write `DatasetManifest`;
> - (4) `train/`: a single `train` CLI wrapping `lerobot-train` per model adapter, logging to W&B;
> - (5) `loop/`: the self-improving controller (EvalReport + coverage map → gap spec → DataEngine request → retrain → eval → promote);
> - (6) `sim/robosimstudio/`: `RoboSimStudioBackend` for sim2sim.
>
> Use only the `contracts/` interfaces to talk to other packages. Write tests first using `DummySim`/`RandomPolicy`.

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| R1 | Foundation contracts + runner (plan Tasks 1–4), pushed to `main` | `contracts/`, `eval/runner.py` | H2 |
| R2 | `configs/robots/fr3_duo.yaml` (joint names, action dim, cameras, control Hz) from provided docs | `configs/robots/` | H2 |
| R3 | `IsaacBackend` with stage flags computed from sim state (predicates in `configs/tasks/handover.yaml`) | `sim/isaac/` | H8 |
| R4 | Ingest: provided dataset → validated LeRobot + manifest `cathlab/handover@v0`; instruction paraphrases; stage segmentation column | `ingest/` | H6 |
| R5 | `train` CLI: `python -m train --model smolvla --data cathlab/handover@v0` → checkpoint + `ModelCard` | `train/` | H8 (v0 run started by H4 on cloud GPU) |
| R6 | Loop controller + learning journal (`reports/loop_journal.md`) | `loop/` | H24 |
| R7 | `RoboSimStudioBackend` for the simplified scene (table + objects + FR3) and sim2sim eval | `sim/robosimstudio/` | H26 |
| R8 | Integration, demo script, final pitch | `scripts/demo.sh`, `docs/pitch/` | H32–36 |

## Interfaces

- **Produces:** `IsaacBackend`, `RoboSimStudioBackend` (both `SimBackend`); `PolicyAdapter` subclasses in `train/adapters/` (`SmolVLAAdapter`, `Pi05Adapter`, `GR00TAdapter`, `ACTAdapter`); datasets `cathlab/handover@v0`, `@v1`, `@v2`; `loop.run_iteration(prev_card: ModelCard) -> ModelCard`.
- **Consumes:**
  - `dataset_intel.coverage_report(dataset_path) -> CoverageReport` (Sharan);
  - `dataengine.generate(gap: GapSpec, n: int) -> DatasetManifest` (Mitesh);
  - `eval.run_suite` + benchmark config (Akanksh);
  - `serve.registry.register(card)` / `promote(card)` (Sharan).

## Notes / commands

- Fine-tune SmolVLA (LeRobot): `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=<hf>/cathlab_handover_v0 --batch_size=64 --steps=20000 --wandb.enable=true`. Check the flags against the installed LeRobot version (`~/lerobot` exists locally).
- RoboSimStudio is installed at `~/robosimstudio`. Its `robosimstudio_real2sim_agent` can scaffold the scene. Coordinate with Mitesh's env factory so both use the same `SceneSpec`.
- Bimanual risk: if a model can't take 16-D actions, decide on single-arm masking in the adapter and record it in `docs/decisions.md`.

## Acceptance

- `uv run pytest` is green; `python -m eval.cli --policy random --sim isaac` produces an `EvalReport`.
- v0, v1 and v2 model cards exist with linked eval reports; the loop journal explains why v2 was promoted.
- The same checkpoint is evaluated in Isaac and RoboSimStudio, and the Δ success is reported.
