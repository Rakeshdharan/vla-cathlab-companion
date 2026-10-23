# Team Workstreams — CathLab Companion

Spec: [`../specs/2026-10-22-cathlab-learning-loop-design.md`](../specs/2026-10-22-cathlab-learning-loop-design.md)
Foundation plan: [`../plans/2026-10-22-cathlab-loop-implementation.md`](../plans/2026-10-22-cathlab-loop-implementation.md)

| Owner | File | Stages / modules |
|---|---|---|
| Rakesh (lead) | [rakesh.md](rakesh.md) | Contracts, shared sim core (`dataengine/`), **robosuite track**, 01 Ingest, 03 training, Loop, RoboSimStudio showcase, integration |
| Sharan | [sharan.md](sharan.md) | **Serving + registry + CI + dataset intelligence + failure report + Docker** ([handoff](../handoffs/2026-10-23-sharan-serve-intel-session.md)) |
| Navneet | [navneet.md](navneet.md) | Awareness / proactive assist, ElevenLabs voice (Isaac Sim dropped at H24) |
| Mitesh | [mitesh.md](mitesh.md) | **ManiSkill3 engine** on the RTX 4090 ([handoff](../handoffs/2026-10-23-rtx4090-maniskill-genesis-session.md)) |
| Akanksh | [akanksh.md](akanksh.md) | VLA research & selection, 04 Evaluate, VLM judge, benchmarking, **Genesis engine** ([handoff](../handoffs/2026-10-23-rtx4090-maniskill-genesis-session.md)) |

## Kickoff: start here (everyone)

> **History was rewritten on 2026-10-22** to clean up commit authorship. If you cloned or pulled before then, run:
> `git fetch && git reset --hard origin/main`

**Step 1: Plan your work (now, before any coding)**

1. Clone and enable the commit hook:
   ```bash
   git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
   git config core.hooksPath .githooks
   ```
2. Read [`AGENTS.md`](../../AGENTS.md) and your own task file (table above).
3. In your agent, run `/start-workstream <your-name>`, or paste the **Agent brief** from your task file plus the spec. The agent writes a test-first plan (superpowers:writing-plans) for your first deliverable to `docs/plans/<date>-<you>-<topic>.md`.
4. Push the plan on branch `feat/<you>-<topic>` and open a PR titled `plan(<you>): <topic>`, so we can review all plans together.
5. **Plan review sync in about 1 h.** Rakesh checks every plan against the spec and the shared interfaces (gaps, name mismatches, overlaps) before anyone starts implementing.

**Step 2: Build (after your plan is approved)**

6. Until `contracts/` is merged (target H2), stick to research and quick experiments. After that, `uv sync && uv run pytest` must be green before you start.
7. Code only inside your own package directories. If you need a contract change, open a dedicated PR touching `contracts/` and tag Rakesh.
8. Small PRs to `main`. Each package needs a `README.md` and at least one test that runs without GPU/simulator (use `DummySim` / `RandomPolicy`). Fill in the PR template checklist.
9. Commits carry **your name only**, with no AI `Co-authored-by` lines (the hook strips them). Log non-obvious choices in [`docs/decisions.md`](../decisions.md) (`/log-decision`).

**Next gates:**
- Plan review in about 1 h.
- H2: contracts merged.
- H6: Akanksh freezes the VLA shortlist and Mitesh freezes the data-engine choice.

## Integration checkpoints (15-min syncs)

| Time | Checkpoint | Gate |
|---|---|---|
| H2 | Contracts merged, env + dataset open | `uv run pytest` green on `main` |
| H6 | VLA shortlist frozen (Akanksh), data-engine choice frozen (Mitesh) | Decision recorded in `docs/decisions.md` |
| H10 | **v0** trained + scored; end-to-end smoke with a real simulator backend | `EvalReport` for scripted expert and v0 in W&B |
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
