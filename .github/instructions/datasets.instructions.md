---
description: "Use when creating, converting, generating, or versioning robot datasets (LeRobot format, synthetic data, MimicGen, domain randomization, Cosmos, teleop) or writing manifest.yaml / DatasetManifest."
---
# Dataset rules

- Every dataset is in **LeRobot format** (parquet + mp4) and has a sibling `manifest.yaml` that validates as `contracts.DatasetManifest`.
- `source` ∈ `provided | domain_rand | mimicgen | scripted | teleop | cosmos`. `parent_versions` lists every dataset it was derived from or mixed with, e.g. `["cathlab/handover@v0"]`.
- Dataset IDs are `cathlab/<task>[_<source>]@v<N>`. Bump `N` for any content change; never overwrite a published version.
- Observation/action keys and dimensions must match `configs/robots/fr3_duo.yaml`. Each episode's `task` string must be a natural-language instruction naming the target object.
- Run `python -m dataset_intel.report <path>` before using a dataset for training, and record its `quality_score` in the manifest.
- Store data on the HF Hub (private) or W&B Artifacts, never in git.
