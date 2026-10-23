---
description: "Use when editing contracts/ (TaskSpec, SceneSpec, DatasetManifest, PolicyAdapter, SimBackend, EvalReport, ModelCard). Rules for changing frozen cross-team interfaces."
applyTo: "contracts/**"
---
# Contract change rules

- `contracts/` is the integration point for all 5 workstreams. Treat every public name, field and signature as frozen.
- Prefer **additive** changes: new optional fields with defaults, new classes. Never rename or remove a field or change a return shape without listing every implementer in the PR.
- In the same PR, update every implementer (`grep -r "PolicyAdapter\|SimBackend" --include=*.py`) and add or adjust tests in `tests/test_contracts.py`.
- Keep `contracts/` dependency-light: only `pydantic`, `numpy`, `pyyaml` and stdlib. No torch, simulator or LeRobot imports.
- Re-export new public types from `contracts/__init__.py` and add them to `__all__`.
- The PR title starts with `feat(contracts):` or `fix(contracts):` and needs Rakesh's review.
