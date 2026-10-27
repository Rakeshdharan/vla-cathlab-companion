# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

The rules above (AGENTS.md) are shared with every coding agent and win on conflict. The notes below add what Claude
needs to work here quickly.

## Commands

```bash
make install            # uv sync (core + dev)
make lint               # uvx ruff check .
make test               # uv run pytest -q: the GPU/simulator/network-free suite CI runs (Python 3.10 and 3.12)
uv run pytest -q tests/serve/test_registry.py::test_candidate_must_beat_production_beyond_ci   # single test
make dev                # Control Station (FastAPI + frontend + harness API) on :8700
make demo-ui            # UI with a throwaway repo and a fake agent
make demo-reset         # rebuild var/registry + production failure report from committed eval reports
uv run python -m harness list                              # agent task templates
uv run python -m harness brief "<request>" --stage <gate>  # deterministic brief for an extension
```

`make lint` runs ruff via `uvx` (pyflakes + whitespace, config in `pyproject.toml`); CI gates on `pytest` only.

Simulators run in **separate Python envs**, not the uv env: conda `robosuite`, `maniskill`, `lerobot`, and a Genesis
venv (`$HOME/venvs/genesis`). Engine tests only run there: `make conformance-{robosuite,maniskill,genesis}`; workers via
`make worker-{robosuite,maniskill,genesis,lerobot}` (ports 8711/8714/8715/8712). Robosuite/ManiSkill need
`MUJOCO_GL=egl`; Genesis takes `CATHLAB_GENESIS_BACKEND=cpu|gpu`.

## Architecture beyond AGENTS.md

- **The closed loop** is `orchestrator/loop.py` (registry, promote/rollback,
  failure reports, targeted next run; mounted under `/api`) + `orchestrator/loop_stages.py` + `report/failure_report.py`
  (`build_report`, `next_run`) + `dataengine.variation.select_configs` (`cells:` split). Promotion rule: `configs/serve.yaml`,
  enforced by `serve/registry.py`.
- **Run flow:** `configs/catalog.yaml` item → `var/runs/<id>/run.yaml` → stages defined in `orchestrator/stages.py`
  (preview → check-expert → generate → convert → train → eval → rollout → register → promote) → each stage shells out
  to an engine CLI on a worker → outputs in `var/runs/<id>/<stage>/<job>/`. State lives in `var/control.db`.
- **Engines are CLIs, not imports.** The orchestrator/UI only know `engines/<id>/engine.yaml` + `python -m
  engines.<id>.cli <stage> --out DIR [--run run.yaml] [--dry-run]` (built with `engines.cli_base.run_cli`). Simulator
  imports stay inside stage functions so `--help` works in the core env. The physics side lives in `sim/<id>/`
  (`SimBackend` + `ExpertSim`), so `dataengine.expert.run_expert_episode` and `engines.expert_policy.ExpertPolicy` run
  unchanged in every engine. Core frame: robot base, table top z = 0, quaternions wxyz; the same `EpisodeConfig` seed is
  the same episode in every engine. Full CLI/stage table: `engines/README.md`.
- **`dataengine/`** is the engine-agnostic sim core: variation specs (`configs/variation/cathlab_v<N>.yaml`, never edit
  `cathlab_v1.yaml`), analytic expert, generators registered with `@register_method`, npz/LeRobot sinks.
- **Harness:** gates in `harness/gates/*.yaml` define contract, conformance command, allowed paths and reference impl per
  extension point; templates in `harness/tasks/*.yaml`. `docs/EXTENDING.md` §2 has the gate table.
- Path-scoped rules for contracts, datasets, engines, models, objects and Python live in `.github/instructions/*.md`;
  read the matching one before editing those areas (e.g. evaluate only via `eval.runner.run_suite` / `python -m eval.cli`
  with a `configs/eval/` suite, report Wilson 95 % CI; tests needing GPU/sim get `@pytest.mark.gpu`; use `logging`, not
  `print`, in library code).

## Commit attribution

AGENTS.md overrides the default: do **not** add `Co-Authored-By` trailers for Claude to commits (the
`.githooks/commit-msg` hook strips them anyway).
