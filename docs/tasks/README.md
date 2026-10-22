# Team Workstreams — CathLab Companion

Spec: [`../specs/2026-10-22-cathlab-learning-loop-design.md`](../specs/2026-10-22-cathlab-learning-loop-design.md)
Foundation plan: [`../plans/2026-10-22-cathlab-loop-implementation.md`](../plans/2026-10-22-cathlab-loop-implementation.md)

| Owner | File | Stages / modules |
|---|---|---|
| Rakesh (lead) | [rakesh.md](rakesh.md) | Contracts, 01 Ingest, Isaac backend, 03 Fine-tune runs, Loop controller (B), RoboSimStudio sim2sim, integration, demo |
| Sharanrajapur | [sharanrajapur.md](sharanrajapur.md) | Literature review + option catalogue, 02 Dataset Intelligence, 05 Version & Serve |
| Navneet | [navneet.md](navneet.md) | Awareness / proactive assist, ElevenLabs voice |
| Mitesh | [mitesh.md](mitesh.md) | DataEngine (synthetic data), env factory, new objects |
| Akanksh | [akanksh.md](akanksh.md) | VLA research & selection, 04 Evaluate, VLM judge, benchmarking |

## How to start (everyone)

1. `git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion`
2. `uv run pytest`. The foundation tests must pass. If `contracts/` is not merged yet, start with research/spikes until H2.
3. `git checkout -b feat/<you>-<topic>`
4. Give your agent the **Agent brief** from your file plus the spec, and ask it to write a TDD plan (superpowers:writing-plans) into `docs/plans/<date>-<you>-<topic>.md` before coding.
5. Code only inside your own package directories. If you need a contract change, open a PR touching `contracts/` and tag Rakesh.
6. Small PRs to `main`. Each package needs a `README.md` and at least one test that runs without GPU/Isaac (use `DummySim` / `RandomPolicy`).

## Integration checkpoints (15-min syncs)

| Time | Checkpoint | Gate |
|---|---|---|
| H2 | Contracts merged, env + dataset open | `uv run pytest` green on `main` |
| H6 | VLA shortlist frozen (Akanksh), data-engine choice frozen (Mitesh) | Decision recorded in `docs/decisions.md` |
| H10 | **v0** trained + scored; end-to-end smoke with real Isaac backend | `EvalReport` for scripted expert and v0 in W&B |
| H14 | First synthetic dataset passes 02 quality gate | `manifest.yaml` + readiness report |
| H20 | **v1** benchmark table (≥2 VLAs) | Table in W&B report |
| H24 | Voice + awareness emit instructions into policy server | Live demo in sim |
| H28 | Loop iteration → **v2**; sim2sim numbers | Loop journal + promotion decision |
| H32 | Feature freeze | Demo videos recorded |
| H36 | Pitch | — |

## Shared rules

- Every dataset has a `manifest.yaml` (`contracts.DatasetManifest`); every checkpoint has a `ModelCard`.
- W&B project: `cathlab-companion`. Name runs `<model>-<dataset_version>-<owner>`.
- Big artifacts (datasets, checkpoints, videos) never go into git. Use the HF Hub (private org) or W&B Artifacts and reference them by URI.
- Secrets (HF token, W&B key, ElevenLabs key) come from `.env` (git-ignored). Never commit them.
- Record decisions in `docs/decisions.md` (one line each: time, decision, why, owner). These feed the pitch.
