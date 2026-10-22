# Decisions Log

One row per non-obvious decision. Newest at the bottom. These feed the pitch.

| Time (H) | Decision | Why | Owner | Links |
|---|---|---|---|---|
| H0 | Contract-first monorepo (approach A) over a full MLOps platform or adopting an existing toolchain wholesale | 5 people in parallel from H2; 36 h budget; keeps our own architecture visible | Rakesh | [spec](specs/2026-10-22-cathlab-learning-loop-design.md) |
| H0 | Headline innovation = self-improving loop (Option B); awareness + voice as add-on | Strongest proof story: success rises v0→v1→v2 | Team | [spec](specs/2026-10-22-cathlab-learning-loop-design.md) |
| H0 | Isaac Sim primary, RoboSimStudio (Newton/MuJoCo-Warp) for sim2sim | Provided env is USD; RoboSimStudio already installed and agent-friendly | Rakesh | |
| H0 | W&B for tracking | LeRobot-native | Team | |
| H0 | Data engines: domain-rand → MimicGen → scripted GraspGen+cuRobo implemented; teleop + Cosmos documented | Robustness and speed first; full catalogue kept for the pitch | Mitesh | [catalogue](specs/2026-10-22-cathlab-learning-loop-design.md#81-data-generation) |
| H6 | **LOCKED — LeRobot-native-first constraint for v0:** SmolVLA (VLA baseline) + ACT (non-VLA lower bound) trained immediately; Diffusion Policy on standby; π0.5 deferred to v1 (openpi adapter needed); GR00T deferred to stretch (Isaac Lab adapter + NVIDIA OML licence) | 36 h budget + dataset already in LeRobot format makes native-first mandatory; non-native models need server/adapter infra not yet built; π0.5 unblocks when Sharan ships serve/client.py | Akanksh | [vla_selection.md](research/vla_selection.md) |
| H1 | Eval suite: 5 tiers × 5 seeds × 6 objects = 150 episodes/model; target < 30 min/model on A100 | Covers in-distribution, new poses, heavy clutter, unseen distractors, new lighting per spec §4 | Akanksh | [suite_v1.yaml](../../configs/eval/suite_v1.yaml) |
| H1 | VLM judge: GeminiJudge (Gemini Vision API) with DummyJudge fallback for CI | Gemini Vision supports video frame analysis; fallback ensures tests pass offline without GPU/API | Akanksh | [judge.py](../../eval/judge.py) |
| H1 | Failure clustering key: (failed_stage, object, scene_variant) → GapSpec with suggested generator heuristic | Enables targeted synthetic data generation per loop spec §6; stage → generator mapping: grasp/lift → scripted, transport/place → mimicgen, reach → domain_rand | Akanksh | [failures.py](../../eval/failures.py) |
