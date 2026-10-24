# ingest

01 Ingest: every dataset (provided or generated) enters the loop through one check.

```bash
uv run --with pyarrow python -m ingest.validate <lerobot_root> --io panda_libero|fr3_duo_16d [--extra <episodes_extra.jsonl>] --out var/ingest/<name>
```

- **Contract:** fps, action dim, state/camera shapes against `engines.registry.IO_PROFILES`; normalisation stats (mean/std/min/max for state and action); at least one language instruction. Exit code 1 if any check fails.
- **Quality:** runs `dataset_intel.report` into `<out>/intel/` and copies its readiness score and issue count.
- Writes `ingest.json` + `ingest.md`. `--no-intel` runs the contract checks without pyarrow.

Instruction paraphrases: `configs/tasks/cathlab_pick.yaml` (3 templates × 3 objects) are used by every generator.
