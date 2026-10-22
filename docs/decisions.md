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
| H6.5 | Robot → single-arm Franka Panda; FR3 Duo + provided USD scene dropped from train/eval (provided data used in 01/02 only) | Most public VLA data/checkpoints are Panda; native in robosuite and Isaac Lab | Rakesh | [sim design](specs/2026-10-22-sim-framework-design.md) |
| H6.5 | Two simulators, Isaac Sim (Isaac Lab) + robosuite, both generate data and both are eval targets, from one shared seeded variation spec | Cross-sim co-training → better held-out generalisation and a smaller sim2sim gap; a measurable claim | Rakesh | [sim design](specs/2026-10-22-sim-framework-design.md) |
| H6.5 | Task "Hand me the {bottle, vial, syringe}" in our own table scene; train grid 3×8 layouts×4 obj colours×4 table colours = 384/sim; held-out layout/colour/table cells for eval | Simple to author identically in both sims; held-out cells prove generalisation | Rakesh | [sim design](specs/2026-10-22-sim-framework-design.md) |
| H6.5 | LIBERO-compatible I/O (7-D EEF delta, 8-D state, agentview + eye_in_hand 256², 20 Hz); fine-tune from LIBERO checkpoints | Reuse pretrained SmolVLA / π0.5 / X-VLA LIBERO heads | Rakesh | [sim design](specs/2026-10-22-sim-framework-design.md) |
| H6.5 | Expert = analytic privileged grasp templates first, GraspGen as expert v2; RoboSimStudio = prompt→scene showcase + optional held-out sim; Cosmos = stretch | Data by H12 without GPU-model risk; avoid a third data pipeline | Rakesh | [sim design](specs/2026-10-22-sim-framework-design.md) |
| H6.5 | Models: SmolVLA (all versions), ACT (v0/v1), one frontier (π0.5 or X-VLA by H10) | Keep the comparison and decide what to drop later | Team | [sim design](specs/2026-10-22-sim-framework-design.md) |
| — | Awareness: next tool comes from the procedure state machine, not the VLM; the VLM only classifies the current step | Keeps proposals restricted to valid task objects and makes the logic testable without a model | Navneet | [plan](plans/2026-10-22-navneet-awareness-engine.md) |
| — | Awareness engine accepts a step only if it stays or advances by one and confidence ≥ 0.6 | One noisy VLM frame must not flip the plan | Navneet | [plan](plans/2026-10-22-navneet-awareness-engine.md) |
| — | Assistant proposes → confirms → emits `instruction`, with cooldown and per-step decline memory; `confirm`/`speak` are injected callables | Avoid nagging; voice (N2) plugs in without changes | Navneet | [plan](plans/2026-10-22-navneet-awareness-engine.md) |
