# vla-cathlab-companion

**Siemens Healthineers Hackathon 2026, Challenge C2:** a simulation-first, self-improving robot learning loop. A Franka FR3 Duo hands cathlab instruments to a clinician on request ("Hand me the guidewire"), or proactively, before being asked.

```
[01 Ingest] → [02 Dataset Intel] → [03 Fine-tune] → [04 Evaluate] → [05 Version & Serve]
     ▲                                                                       │
     └──── DataEngine ◄── Loop Controller (failures → gaps → new data) ◄─────┘
```

| Doc | What |
|---|---|
| [docs/specs/…-design.md](docs/specs/2026-10-22-cathlab-learning-loop-design.md) | Architecture, stages, option catalogue, timeline |
| [docs/plans/…-implementation.md](docs/plans/2026-10-22-cathlab-loop-implementation.md) | Foundation implementation plan (contracts + eval runner) |
| [docs/tasks/](docs/tasks/README.md) | Per-member workstreams and integration checkpoints |
| [docs/decisions.md](docs/decisions.md) | Decisions log |
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
