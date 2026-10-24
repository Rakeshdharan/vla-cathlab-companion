# vla-cathlab-companion

**Siemens Healthineers Hackathon 2026, Challenge C2:** a simulation-first, self-improving robot learning loop. A Franka FR3 Duo hands cathlab instruments to a clinician on request ("Hand me the guidewire"), or proactively, before being asked.

```
[01 Ingest] → [02 Dataset Intel] → [03 Fine-tune] → [04 Evaluate] → [05 Version & Serve]
     ▲                                                                       │
     └──── DataEngine ◄── Loop Controller (failures → gaps → new data) ◄─────┘
```

## Control Station (UI + agent harness)

```bash
make install && make test      # core env + GPU-free tests
make dev                       # http://localhost:8700  (UI, catalog, runs, agent harness)
make worker-robosuite          # robosuite worker :8711 (needs the robosuite conda env)
make demo-ui                   # try the UI with a throwaway repo and a fake agent
```

- **PROJECT:** pick simulator, embodiment, objects, expert, variation and model from the catalog. Anything not ready offers **Build with agent**.
- **PIPELINE:** pick a run, run stages (preview → check expert → generate → convert → train → eval → rollout → register → promote) on the workers or the control plane, follow logs and outputs.
- **MODELS:** register evaluated checkpoints as versions (`cathlab-pick@vN`), promote by the CI rule, roll back.
- **REPORT:** failure report (failing cells, sim2sim gap, recommendations) → CREATE NEXT RUN (targeted generate).
- **ASSIST:** the procedure step proposes the object; YES hands it over with the production model.
- **AGENT:** run a task template with a coding agent (GitHub Copilot CLI by default) in an isolated git worktree, then review the diff and push the branch. See [docs/HARNESS.md](docs/HARNESS.md).
- **AGENT BRIEF:** describe anything the UI doesn't offer; get a brief for any coding agent.
- Install (native, remote GPU worker): [docs/INSTALL.md](docs/INSTALL.md) · Credits: [CREDITS.md](CREDITS.md).

| Doc | What |
|---|---|
| [docs/specs/…-design.md](docs/specs/2026-10-22-cathlab-learning-loop-design.md) | Architecture, stages, option catalogue, timeline |
| [docs/plans/…-implementation.md](docs/plans/2026-10-22-cathlab-loop-implementation.md) | Foundation implementation plan (contracts + eval runner) |
| [docs/tasks/](docs/tasks/README.md) | Per-member workstreams and integration checkpoints |
| [docs/decisions.md](docs/decisions.md) | Decisions log |
| [docs/DEMO.md](docs/DEMO.md) | 5-minute Control Station demo walk-through, with fallbacks |
| [AGENTS.md](AGENTS.md) | Rules for coding agents (and humans) working in this repo |

## Quick start

> **Team: start with the [Kickoff steps](docs/tasks/README.md#kickoff-start-here-everyone)** — plan first, then build.

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks   # strips AI co-author trailers from commits
uv sync && uv run pytest
```

Agent workflows live in `.github/`:
- Prompts: `/start-workstream`, `/new-policy-adapter`, `/new-sim-backend`, `/new-data-generator`, `/log-decision`.
- Scoped instructions for contracts, Python, datasets and models.
- A `Contract Reviewer` agent for PRs.

Team: Rakesh (lead), Navneet, Mitesh, Akanksh, Sharan.
