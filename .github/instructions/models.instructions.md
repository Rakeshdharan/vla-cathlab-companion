---
description: "Use when training, fine-tuning, evaluating, comparing, promoting or serving VLA models (SmolVLA, pi0, pi0.5, GR00T, X-VLA, ACT) or writing ModelCard / EvalReport."
---
# Model training, evaluation and versioning rules

- Wrap every model as a `contracts.PolicyAdapter` (LeRobot policies: `train/lerobot_adapter.py`; recipes in `train/recipes/<model>.yaml`). `predict` returns `(chunk_size, action_dim)` in the robot's action space; normalisation happens inside the adapter.
- Every training run logs to W&B project `cathlab-companion` with run name `<model>-<dataset_version>-<owner>`, and writes a `ModelCard` YAML (dataset versions, W&B run, git SHA, checkpoint URI).
- Evaluate only through `eval.runner.run_suite` / `python -m eval.cli` with a suite from `configs/eval/`. Fixed seeds, no ad-hoc scripts for reported numbers.
- Report success with its Wilson 95% CI, the failure-stage histogram and latency p50/p95. Never compare models evaluated on different suites.
- Promotion (`serve.registry.promote`) only when the rule in `configs/serve.yaml` passes. Record promotions and rejections (and the evidence) in the PR description.
