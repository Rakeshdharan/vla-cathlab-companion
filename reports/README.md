# reports/

Committed evaluation results, as plain files so they can be read without running anything.

| Path | What |
|---|---|
| `robosuite/eval_reports/<engine>/<model>/eval_report.json` | One `contracts.EvalReport` per (engine, checkpoint) cell: per-episode results (success, failed stage, latency), used by `scripts/demo_bootstrap.py` to rebuild the registry and by the cross-physics report below |
| `robosuite/cross_physics_v1.md` / `.json` | The cross-physics table (every model × every eval engine, Wilson 95 % CI, per-tier breakdown, failed-stage/latency summary); see the top-level [README.md](../README.md#results) for the headline numbers |

## How these are produced

1. **Eval:** run each checkpoint through every engine's `eval` stage, writing `eval_report.json`:
   ```bash
   python -m engines.robosuite.cli eval --policy lerobot:<ckpt> --suite configs/eval/suite_v2.yaml \
       --out var/eval/robosuite/<ckpt>
   ```
   Copy (or symlink) the resulting `eval_report.json` under `robosuite/eval_reports/<engine>/<ckpt>/` to commit it.
2. **Cross-physics table:** once `reports/robosuite/eval_reports/<engine>/<model>/eval_report.json` exist for every
   (engine, model) cell you want in the table:
   ```bash
   python -m train.cross_physics_report --evals reports/robosuite/eval_reports --out reports/robosuite/cross_physics_v1
   ```
   This writes `cross_physics_v1.md` (the table) and `cross_physics_v1.json` (machine-readable summary).
3. **Registry / promotion:** `scripts/demo_bootstrap.py` reads `reports/robosuite/eval_reports/robosuite/<ckpt>/eval_report.json`
   for every checkpoint in its `CHECKPOINTS` list and registers a version for each (`make demo-bootstrap` /
   `make demo-reset`).

Model and dataset ids in the table come from `train/cross_physics_report.py`'s `MODELS` mapping; add a new
checkpoint there (and its eval reports under `robosuite/eval_reports/`) before regenerating.
