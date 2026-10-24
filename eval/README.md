# eval/ — CathLab Evaluation Harness

## Modules

| File | What |
|---|---|
| `runner.py` | `run_episode` / `run_suite` (frozen API) |
| `metrics.py` | Wilson CI, by-tier, by-object, latency, grounding |
| `judge.py` | VLM success judge (DummyJudge + GeminiJudge) |
| `cli.py` | `python -m eval.cli --policy X --sim Y --suite Z` |
| `benchmark.py` | Benchmark table + matplotlib plots |
| `failures.py` | Failure clustering → GapSpec (feeds `report/`) |

## Quick Start

```bash
# Smoke test — no GPU, no API keys needed
python -m eval.cli --policy random --sim dummy --suite suite_v1 --model-version random@v0

# Full eval on the GPU workstation
python -m eval.cli \
  --policy ws://workstation:8765 \
  --sim robosimstudio \
  --suite suite_v1 \
  --model-version smolvla@v0 \
  --dataset-versions "cathlab/handover@v0" \
  --wandb \
  --vlm-judge

# Generate benchmark table from saved reports
python -m eval.benchmark --reports reports/ --out reports/benchmark.md
```

## Extending the Runner

`run_episode` and `run_suite` signatures are **frozen** (spec §4). Add new behaviour by:
- Video capture: hook into `cli.py` before/after `run_episode`
- VLM judge: `judge.py` → verdicts attached in `cli.py._attach_judge_verdicts()`
- New metrics: add functions to `metrics.py`
- New tiers / seeds: edit `configs/eval/suite_v1.yaml`

Never import `eval` internals from other packages — import only from `contracts`.

## Eval Suite

[`configs/eval/suite_v1.yaml`](../configs/eval/suite_v1.yaml):
- **5 difficulty tiers** × 5 seeds × 6 objects = **150 episodes / model**
- Target runtime: **< 30 min / model** on a single 40 GB GPU

| Tier | Scene Variant | Challenge |
|---|---|---|
| `in_distribution` | default | Standard training distribution |
| `new_poses` | new_poses | Objects rotated / shifted ±15 cm, ±45° |
| `heavy_clutter` | heavy_clutter | 5 random distractor objects on table |
| `unseen_distractors` | unseen_distractors | Novel objects not seen in training |
| `new_lighting` | new_lighting | Intensity and colour temperature varied |

## Metrics

| Metric | Where |
|---|---|
| Success rate + Wilson 95% CI | `EvalReport.success_rate()`, `.wilson_ci()` |
| Failure stage histogram | `EvalReport.stage_failure_hist()` |
| Success by tier / object | `metrics.success_by_tier()`, `.success_by_object()` |
| Time-to-success (mean, median, p95 steps) | `metrics.time_to_success()` |
| Inference latency p50 / p95 | `metrics.latency_summary()` |
| Language grounding accuracy | `metrics.language_grounding_accuracy()` |
| Collision rate | `metrics.collision_rate()` |

## VLM Judge

`judge.py` provides two implementations behind the same interface:

| Class | When used | API key |
|---|---|---|
| `DummyJudge` | Tests / offline / CI | None |
| `GeminiJudge` | Real eval with `--vlm-judge` | `GOOGLE_API_KEY` in `.env` |

`agreement(verdicts, ground_truths)` measures judge vs sim ground-truth — reported in the `EvalReport`.

## Failure Analysis → Loop Controller

```python
from contracts import EvalReport
from eval.failures import FailureAnalyser

report = EvalReport.load("reports/smolvla-v0.json")
gaps = FailureAnalyser(report).to_gaps()
# gaps is list[GapSpec] sorted by priority, consumed by loop/
```

## Tests

```bash
uv run pytest tests/test_contracts.py tests/test_runner.py tests/test_metrics.py \
              tests/test_judge.py tests/test_failures.py -v
```

All tests use `DummySim` and `RandomPolicy`; no GPU, simulator, or API keys required.
