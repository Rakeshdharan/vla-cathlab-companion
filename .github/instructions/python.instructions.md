---
description: "Python coding and testing conventions for this repo: typing, pydantic configs, lazy heavy imports, GPU-free tests."
applyTo: "**/*.py"
---
# Python conventions

- Python ≥3.10, type hints on public functions, `pydantic` models for anything loaded from YAML/JSON.
- Import heavy or optional dependencies (`torch`, `lerobot`, `isaacsim`, `omni.*`, `curobo`, `elevenlabs`, `wandb`) **inside** the function or class that needs them, so `uv run pytest` works on a laptop without a GPU.
- Tests live in `tests/` (or `<package>/tests/`). They use `DummySim`, `RandomPolicy`, `tmp_path` and tiny fixtures, never the network, real datasets or a GPU. Mark GPU/sim tests with `@pytest.mark.gpu` and skip them by default.
- Read paths, seeds, object lists and hyperparameters from `configs/`, never hard-code them.
- No bare `print` in library code; use `logging.getLogger(__name__)`.
