---
name: cathlab-extend
description: Extend the CathLab Companion platform (new engine, object, data method, metric, report section, or any pipeline stage). Use when asked to add or change a simulator, object, synthetic data method, eval metric, rollout or report feature in this repo.
---
# Extending CathLab Companion

1. Find the stage gate in `harness/gates/` (engine, scene, objects, expert, data_method, convert, intel, train, eval,
   rollout, assist, report). Read its contract, reference implementation and context files.
2. Generate the brief: `python -m harness brief "<request>" --stage <gate> [--engine <id>]`; follow it.
3. If a harness template matches (`python -m harness list`), prefer `python -m harness run <template> --param ...`.
4. Run the gate's conformance command and `uv run pytest -q`; update status flags only for what now works.
5. Rules: AGENTS.md; human-only commit authorship (no AI Co-authored-by); no data or secrets in git; CREDITS.md and
   docs/decisions.md rows.
