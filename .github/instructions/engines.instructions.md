---
applyTo: "engines/**,sim/**,tests/engines/**"
---
# Engine plugins

- Follow `harness/gates/engine.yaml` and `docs/EXTENDING.md`; mirror `engines/robosuite/`.
- Build every CLI with `engines.cli_base.run_cli`; import simulators lazily so `--help` works in the core env.
- Exit 0 on success, non-zero with a one-line reason on stderr; write only under `--out`.
- Core frame: robot base, table top z = 0, quaternions wxyz. The same EpisodeConfig seed is the same episode everywhere.
- Mark a capability `ready` in `engine.yaml` only after its check passes, and log the evidence in `docs/decisions.md`.
- Conformance: `python -m pytest -q tests/engines/test_conformance.py -k <engine>` in the engine's env.
