# dataset_intel

Validators, train-grid coverage and a readiness score for LeRobot v3 datasets (spec Part B S5).

```bash
uv run --with pyarrow python -m dataset_intel.report <lerobot_root> [--extra <root>/meta/episodes_extra.jsonl] \
    [--io panda_libero|fr3_duo_16d] [--spec configs/variation/cathlab_v1.yaml] --out var/intel/<name>
```

| Module | What |
|---|---|
| `model.py` | `IntelReport` (dataset, issues, coverage, readiness): the `report.json` schema |
| `validators.py` | `nan`, `frozen`, `action_jump`, `length_outlier`, `gripper_never_closes`, `idle_fraction`, `frame_count_mismatch` (numpy only) |
| `coverage.py` | Counts per (object, layout, obj colour, table colour, source sim) and empty `cathlab_v1` train cells, from `episodes_extra.jsonl` |
| `report.py` | Loads a v3 root with pyarrow, runs the validators, scores readiness, writes `report.json` + `report.md` |

Thresholds and readiness weights: `configs/intel.yaml`. Readiness is the weighted mean of the components that apply
(valid fraction, coverage fraction and object balance need `--extra`, and 1 − mean idle fraction). Gripper conventions:
`panda_libero` closed when `action[6] > 0`; `fr3_duo_16d` right gripper `action[15] < 0.5` (1 = open). Both FR3 gripper
dims (7, 15) are excluded from `action_jump`.
