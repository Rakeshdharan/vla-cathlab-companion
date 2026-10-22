# Akanksh — Frontier VLA Research, Evaluation Design, Benchmarking

> **Update (H6.5):** see the [simulation framework spec](../specs/2026-10-22-sim-framework-design.md).
> - **Models:** SmolVLA (v0/v1/v2), ACT (v0/v1), and ONE frontier model, π0.5 or X-VLA. Decide by **H10**, all starting from public **LIBERO** checkpoints.
> - **Eval suite → `configs/eval/suite_v2.yaml`:** per simulator, 3 objects × 5 tiers × 4 seeds = 60 episodes. Generate the configs with `dataengine.variation.eval_configs(spec, tier, seeds)` and `TIERS` (in_distribution, held_out_layout, held_out_obj_color, held_out_table_color, all_held_out), with seeds ≥ 5000.
> - **Backends:** `IsaacBackend` (Mitesh), `RobosuiteBackend` (Rakesh), `FakeExpertSim` for tests.
> - **Report** trained vs held-out cells, the Isaac↔robosuite gap, and wrong-object as a failure class.
> - **Benchmark:** {expert, SmolVLA v0/v1/v2, ACT v0/v1, frontier v1} × {Isaac, robosuite} × {trained, held-out}.

## Agent brief (paste to your agent)

> You are helping build the CathLab Companion learning loop (spec: `docs/specs/2026-10-22-cathlab-learning-loop-design.md`, §4 "04 Evaluate", §6, §8.2).
>
> My scope:
> - (1) research and shortlist VLAs for bimanual FR3 Duo pick-and-place with language grounding (SmolVLA, pi0/pi0.5, GR00T N1.x, X-VLA, OpenVLA-OFT, plus ACT/Diffusion Policy as non-VLA baselines), ranked by bimanual support, LeRobot/openpi integration, fine-tune cost on ≥40 GB GPUs, inference latency, and licence;
> - (2) design and implement the evaluation suite in `eval/`: seeded scenes × objects × difficulty tiers (in-distribution, new poses, heavy clutter, unseen distractors, lighting), metrics (success with Wilson CI, failure-stage histogram, time-to-success, latency p50/p95, collisions, language-grounding accuracy = picked the correct object), and video capture;
> - (3) a VLM success judge that explains where a rollout degraded, validated against sim ground-truth flags;
> - (4) the benchmark table and plots comparing models × data versions × simulators.
>
> Extend `eval/runner.py` without changing the `run_episode`/`run_suite` signatures.

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| A1 | VLA shortlist + decision (top 2–3 to fine-tune, plus baselines) with reasoning and adapter notes for Rakesh | `docs/research/vla_selection.md`, `docs/decisions.md` | H6 |
| A2 | Eval suite config: tiers, seeds, objects, episode budget (sized so a full eval takes <30 min per model) | `configs/eval/suite_v1.yaml` | H8 |
| A3 | `eval/cli.py`: `python -m eval.cli --policy <adapter\|url> --sim <backend> --suite suite_v1` → `EvalReport` + videos + W&B logging | `eval/cli.py`, `eval/metrics.py` | H10 |
| A4 | VLM judge: `judge(video_path, instruction) -> Verdict(success, failed_stage, explanation)`; agreement % vs ground truth reported | `eval/judge.py` | H18 |
| A5 | Benchmark: {ScriptedExpert, ACT, SmolVLA, 1–2 frontier VLAs} × {v0 provided, v1 + synthetic, v2 + loop} × {Isaac, RoboSimStudio} → table + plots (success w/ CI, stage failures, latency) | `eval/benchmark.py`, `reports/benchmark.md` | H20 (v1), H28 (final) |
| A6 | Failure analysis feeding the loop: cluster failed episodes into candidate `GapSpec`s | `eval/failures.py` | H22 |

## Interfaces

- **Consumes:** `PolicyAdapter` implementations (Rakesh, Mitesh's `ScriptedExpert`, Sharan's `PolicyClient`); `SimBackend`s; `ModelCard`.
- **Produces:** `EvalReport` JSONs in `reports/`, `Verdict`, benchmark artifacts, `failures.to_gaps(report) -> list[GapSpec]`.

## Acceptance

- Every model version has a reproducible `EvalReport` (fixed seeds, logged config, git SHA).
- The benchmark shows v0 → v1 → v2 with CIs and an honest sim2sim gap.
- The VLM judge's agreement with ground truth is reported, and its explanations appear in the demo.
