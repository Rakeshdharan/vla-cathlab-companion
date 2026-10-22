---
description: "Kick off your workstream: read your task file and write a TDD implementation plan before coding."
argument-hint: "owner name, e.g. mitesh"
agent: "agent"
---
Start my workstream for this hackathon repo.

1. Read [AGENTS.md](../../AGENTS.md), the [design spec](../../docs/specs/2026-10-22-cathlab-learning-loop-design.md), and my task file `docs/tasks/${input:owner}.md`.
2. Check that the foundation exists (`contracts/`, `eval/runner.py`) and that `uv run pytest` is green. If it doesn't exist yet, say so and limit the work to research/spikes.
3. Create the branch `feat/${input:owner}-<first-deliverable-topic>`.
4. Write a bite-sized TDD implementation plan for my **next undone deliverable** to `docs/plans/<YYYY-MM-DD>-${input:owner}-<topic>.md`. Use exact file paths, the interfaces from my task file, test code, and run commands.
5. Stop and show me the plan before implementing.
