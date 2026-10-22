# Decisions Log

One row per non-obvious decision. Newest at the bottom. These feed the pitch.

| Time (H) | Decision | Why | Owner | Links |
|---|---|---|---|---|
| H0 | Contract-first monorepo (approach A) over a full MLOps platform or adopting an existing toolchain wholesale | 5 people in parallel from H2; 36 h budget; keeps our own architecture visible | Rakesh | [spec](specs/2026-10-22-cathlab-learning-loop-design.md) |
| H0 | Headline innovation = self-improving loop (Option B); awareness + voice as add-on | Strongest proof story: success rises v0→v1→v2 | Team | [spec](specs/2026-10-22-cathlab-learning-loop-design.md) |
| H0 | Isaac Sim primary, RoboSimStudio (Newton/MuJoCo-Warp) for sim2sim | Provided env is USD; RoboSimStudio already installed and agent-friendly | Rakesh | |
| H0 | W&B for tracking | LeRobot-native | Team | |
| H0 | Data engines: domain-rand → MimicGen → scripted GraspGen+cuRobo implemented; teleop + Cosmos documented | Robustness and speed first; full catalogue kept for the pitch | Mitesh | [catalogue](specs/2026-10-22-cathlab-learning-loop-design.md#81-data-generation) |
| H2 | Provided datasets + env hosted as private HF dataset repos (not Git LFS) | 5.6 GB exceeds GitHub's free LFS quota; LeRobot loads HF repos natively | Rakesh | [data.md](data.md) |
| H2 | Provided data = box handover (94 eps, 30 fps) + cube stacking (490 eps, 6 fps), one fixed instruction each. Cathlab "hand me the &lt;object&gt;" data must come from the DataEngine; instruction enrichment is mandatory | Found while validating the datasets in LeRobot | Rakesh | [data.md](data.md) |
