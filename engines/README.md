# `engines/` — simulation engine plugins

An **engine plugin** is one simulator wired into the platform behind a single contract: a folder `engines/<id>/`
with an `engine.yaml` manifest and a `cli.py` exposing six stages, plus a `SimBackend` and an `ExpertSim` adapter in
`sim/<id>/`. The orchestrator, the UI and agents never special-case a simulator; they only call the manifest and the
CLI. Today's plugins: `engines/robosuite/` (MuJoCo 3.3, Rakesh) and `engines/maniskill/` (ManiSkill3 / PhysX 5,
Mitesh), both `panda_libero` I/O. `engines/genesis/` is planned (Akanksh). Isaac Sim was dropped. MuJoCo/Newton for
the FR3 Duo embodiment live on Aravind's own repo track.

## CLI usage

From `engines/cli_base.py`'s docstring:

```
python -m engines.<id>.cli <stage> --out DIR [--run var/runs/<id>/run.yaml] [stage flags] [--dry-run]
```

Contract (`docs/EXTENDING.md`): exit 0 on success; non-zero with a one-line reason on stderr; progress on stdout;
outputs only under `--out`. `--dry-run` prints each command a stage would run (one JSON argv per line), runs nothing.

## Stages

| Stage | Flags | Outputs in `--out` |
|---|---|---|
| `preview` | `--n` (train configs in the grid), `--spec` | `grid.png` (n train configs + 1 all-held-out), `preview.json` |
| `check-expert` | `--n`, `--split`, `--objects`, `--param k=v` (repeatable), `--spec` | `expert_check.json`: `{split, params, per_object: {obj: {success, attempted, rate, mean_steps, max_steps}}, failures}` |
| `generate` | `--split`, `--start`, `--n`, `--shard`, `--num-shards`, `--dataset-id`, `--version`, `--created-by`, `--spec` | `raw/shard_<i>-of-<k>/episodes/*.npz`, `meta/episodes_extra.jsonl`, `manifest.yaml` |
| `convert` *(run in the lerobot env)* | `--dataset` (the `generate` stage's `--out`), `--repo-id` | `lerobot/` (LeRobot v3), tags `v<N>` + `v3.0` when pushed |
| `eval` | `--suite`, `--policy` (`expert`\|`random`\|`url:<policy server>`), `--model-version`, `--tiers`, `--max-seeds`, `--video` (`none`\|`failures`\|`all`), `--objects`, `--spec` | `eval_report.json` (`contracts.EvalReport`), `videos/` |
| `rollout` | `--policy`, `--object`, `--seed`, `--tier`, `--instruction`, `--max-steps`, `--spec` | `rollout.mp4`, `rollout.json` (`{success, steps, failed_stage, latency_ms_p50, latency_ms_p95}`) |

See `docs/EXTENDING.md` for the full engine contract, the stage-gate table and how to add a new engine, object, data
method, metric or report section.
