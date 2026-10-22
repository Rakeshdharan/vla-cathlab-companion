---
description: "Scaffold a new synthetic data generator for the DataEngine that accepts a GapSpec and emits a LeRobot dataset + manifest."
argument-hint: "method name, e.g. domain_rand, mimicgen, scripted, teleop, cosmos"
agent: "agent"
---
Add a DataEngine generator `${input:method}`.

- File: `dataengine/${input:method}.py`, registered in `dataengine/__init__.py` so that `dataengine.generate(gap, n, method="${input:method}") -> DatasetManifest` works.
- Honour the `GapSpec` (object, region, clutter, stage) when it is given; otherwise sample the full task distribution from `configs/tasks/` and `configs/scenes/`.
- Output a LeRobot dataset plus `manifest.yaml` that follows the dataset rules (`source: ${input:method}`, `parent_versions`, versioned ID).
- Test with `DummySim` and n=2 episodes, checking the manifest fields and episode count; real generation is marked `@pytest.mark.gpu`.
- Update the option catalogue in `docs/research/data_engines.md` (status, speed, robustness, cost).
