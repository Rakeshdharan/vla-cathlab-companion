---
tags: [index]
---

# CathLab Companion: knowledge base

Index for the repo as an [Obsidian](https://obsidian.md) vault (open the repo root as a vault). Links are plain
Markdown, so every note also renders on GitHub.

## Start here

- [README](../README.md): what this is, results, datasets, checkpoints
- [Install](INSTALL.md): native and remote-GPU-worker setup
- [Control Station guide](GUIDE.md): every tab, button and pipeline stage, troubleshooting
- [Demo walk-through](DEMO.md): 5 minutes, REPORT → MODELS → NEXT RUN → ASSIST

## How it fits together

- [Extending the platform](EXTENDING.md): run flow, stage gates, adding an engine / object / data method / metric
- [Agent harness](HARNESS.md): task templates, worktrees, briefs
- [Rules for contributors and agents](../AGENTS.md)
- [Credits](../CREDITS.md)

## Packages

| Stage | Note |
|---|---|
| Engines (uniform 6-stage CLI) | [engines](../engines/README.md) |
| Simulator adapters | [robosuite](../sim/robosuite/README.md) · [ManiSkill3](../sim/maniskill/README.md) · [Genesis](../sim/genesis/README.md) |
| 01 ingest | [ingest](../ingest/README.md) |
| 02 dataset intelligence | [dataset_intel](../dataset_intel/README.md) |
| 03 train | [train](../train/README.md) |
| 04 eval | [eval](../eval/README.md) |
| ASSIST | [awareness](../awareness/README.md) |

## Data and results

- [Datasets](data.md): provided and generated, formats, access, quality rules
- [Reports](../reports/README.md) and the [cross-physics table](../reports/robosuite/cross_physics_v1.md)

![Cross-physics success heatmap](assets/plots/cross_physics_heatmap.png)

## Media

Figures are generated, not drawn by hand:

| Asset | Source | Regenerate |
|---|---|---|
| `assets/plots/*.png` | `reports/robosuite/cross_physics_v1.json` | `uv run --with matplotlib python -m scripts.plot_results` |
| `assets/gifs/dataset_*.gif` | first episodes of the HF `pick_*` datasets | ffmpeg from the LeRobot videos (see the README's Datasets section) |
| `assets/gifs/rollout_*.gif` | `configs/demo_videos.yaml` recordings | ffmpeg |
| `assets/ui/*.png` | the Control Station after `make demo-reset` | screenshots of `make dev` |
