# CathLab Companion — Autonomous Robot Learning Loop (Design Spec)

*Siemens Healthineers Hackathon 2026, Challenge C2 · Team: Rakesh (lead), Navneet, Mitesh, Akanksh, Sharan · 36 h, agent-assisted*

## 1. Goal

Build a **repeatable, simulation-first learning loop** that turns demonstrations into a served, evaluated and versioned robot capability, and then improves itself.

Reference task: *"Hand me the `<object>`"*. The FR3 Duo picks the named cathlab item from a cluttered instrument table and places it on a handover table.

**Definition of done (from the challenge):**
- A fine-tuned VLA checkpoint.
- A demonstrable stack covering 01 Ingest → 02 Dataset Intelligence → 03 Fine-tune → 04 Evaluate → 05 Version & Serve.
- Task execution shown with our own expert policy.

**Our headline (Option B: Self-Improving Pipeline):** evaluation failures automatically drive targeted synthetic data generation and retraining. We prove the loop with a rising success rate across versions v0 → v1 → v2.

**Add-ons:**
- Option A: VLM success judge.
- Option D/X: proactive "awareness" assist. The robot anticipates the next tool from scene and procedure context, and also accepts voice requests (ElevenLabs).
- Prompt-to-environment creation, to show the pipeline generalises to new tasks and scenes.

**Out of scope:**
- Navigation.
- Real-robot deployment. The design is real-ready through the policy server, but it is not demoed on hardware.

## 2. Given inputs and constraints

| Item | Value |
|---|---|
| Robot | Franka FR3 Duo (bimanual). Action/state `float32[16]` = 2 × (7 joints + gripper); 3 cameras (external, left/right wrist) at 224×224 |
| Dataset | LeRobot v3.0, programmed controllers: box handover 10 + 84 eps (30 fps) and cube stacking 490 eps (6 fps), one fixed instruction each. See [docs/data.md](../data.md) |
| Environment | Clinical 3D scene, **Isaac Sim / USD** |
| Compute | Cloud GPUs (≥40 GB) for training; laptops (8 GB) for dev and inference |
| Tracking | **Weights & Biases** |
| Time | 36 h, 5 people, each with coding agents, one shared git repo |

## 3. Architecture (contract-first monorepo)

```
             ┌──────────── Loop Controller (Option B) ─────────────────────────────┐
             │ eval report → failure clustering → gap spec → DataEngine request →   │
             ▼ retrain → re-eval → promote if better                               │
[01 Ingest] → [02 Dataset Intel] → [03 Fine-tune] → [04 Evaluate] → [05 Version & Serve]
     ▲                                  PolicyAdapter     SimBackend        registry + policy server
     │                                                                            │
[DataEngine] ◄── [Scene/Env Factory]                       [Awareness + Voice front-end] ─┘
```

**Principles:**
- Stages communicate **only through versioned artifacts** that carry a `manifest.yaml`.
- Every model sits behind `PolicyAdapter`, and every simulator sits behind `SimBackend`.
- Adding a model, sim, data source or task is a plugin, not a rewrite. This is the "any task" claim.

### 3.1 Frozen contracts (H0–2, owner: Rakesh)

| Contract | Shape |
|---|---|
| `TaskSpec` (YAML) | `task_id`, instruction templates (`"hand me the {object}"`), object set, success predicate, stage predicates (reach/grasp/lift/transport/place) |
| `SceneSpec` (YAML) | USD/MJCF asset refs, object poses or randomization ranges, cameras, lighting |
| Dataset | LeRobot v2/v3 format plus `manifest.yaml`: `source` (provided / domain_rand / mimicgen / scripted / teleop / cosmos), `parent_versions`, `task_id`, `scene_id`, `n_episodes`, `quality_score` |
| `PolicyAdapter` | `load(ckpt)`, `predict(obs: dict, instruction: str) -> action_chunk`, `meta()` (name, params, chunk size, Hz) |
| `SimBackend` | `reset(scene_spec, seed) -> obs`, `step(action) -> obs, info`, `stage_flags()`, `render()` |
| `EvalReport` (JSON) | per-episode success, failed stage, seed, scene variant, latency p50/p95, video path, VLM-judge verdict; aggregates per model/version |
| `ModelCard` | registry entry: version, dataset versions, W&B run, eval report, git SHA |

### 3.2 Repo layout

```
contracts/        schemas + base classes (PolicyAdapter, SimBackend, specs)
configs/          tasks/, scenes/, models/, experiments/
ingest/           01: loaders, normalization, schema conversion, enrichment (language variants)
dataset_intel/    02: validators, dedup, coverage/gap maps, VLA-readiness report
dataengine/       synthetic generators: domain_rand, mimicgen, scripted_expert, teleop, cosmos
envfactory/       prompt→scene, manual scene builder, new object/asset creation
sim/isaac/        IsaacBackend (primary)
sim/robosimstudio/  RoboSimStudioBackend (sim2sim, Newton/MuJoCo-Warp)
train/            03: LeRobot training wrappers per model, configs
eval/             04: harness, metrics, VLM success judge, benchmark tables
serve/            05: registry, policy server (WebSocket), rollout CLI, Docker
loop/             Option B controller
awareness/        proactive assist + ElevenLabs voice front-end
docs/             specs, plans, literature review, option catalogue, pitch
```

## 4. Stage design

### 01 Ingest & Transform
- Load the provided LeRobot dataset, check it against the contract, and normalise action/state stats.
- **Enrich:**
  - paraphrased instructions per episode (LLM), so the policy is robust to language;
  - object-name labels;
  - stage segmentation (reach/grasp/lift/transport/place) from gripper and object-state traces.
- Synthetic sources enter through the same path, so every dataset is LeRobot plus a manifest.

### 02 Dataset Intelligence
- **Validity checks:** NaN/frozen frames, action jumps, gripper never closes, object never lifted, length outliers, camera desync.
- **Redundancy:** near-duplicate episodes (trajectory DTW plus image embeddings).
- **Coverage map:** object × table-region × clutter-level × instruction. Empty cells are the **gap spec** consumed by the loop.
- **Output:** an HTML/markdown readiness report with a VLA-readiness score, logged to W&B.

### 03 Fine-tune
- One training entrypoint: `train --model <adapter> --data <dataset@version>`.
- Model selection is owned by Akanksh (§6). Candidates are listed in the catalogue (§8.2).
- **Default plan:**
  - SmolVLA = fast baseline (trains in hours, cheap inference);
  - pi0.5 and/or GR00T N1.x = frontier contenders;
  - ACT = non-VLA lower bound.

### 04 Evaluate
- **Fixed eval suite:** N seeded scenes × object set × difficulty tiers (in-distribution, new poses, heavy clutter, unseen distractors, new lighting).
- **Metrics:**
  - success rate with 95% CI;
  - **failure stage distribution**;
  - time-to-success;
  - inference latency p50/p95;
  - collisions;
  - language grounding accuracy (picked the *right* object).
- **VLM success judge (Option A):** reads the rollout video and explains where execution degraded. It is checked against ground-truth sim predicates.
- **Sim2sim:** the same suite runs in RoboSimStudio. The sim2sim gap is reported as Δ success.

### 05 Version & Serve
- **Registry:** a versioned `ModelCard` linking dataset versions, W&B run, eval report and git SHA. Storage is the HF Hub (private) or W&B Artifacts.
- **Policy server:** WebSocket/gRPC, `predict(obs, instruction)`, following the openpi/LeRobot async-inference pattern. The same server drives Isaac, RoboSimStudio or a real robot.
- **Versioned rollout:**
  - `rollout --model cathlab-pick@v2 --sim isaac|robosimstudio`;
  - promotion rule: v(n+1) is promoted only if its success rate beats v(n) beyond the CI and latency stays within budget;
  - one-command rollback.

### Loop Controller (Option B, headline)
1. Read `EvalReport` and the 02 coverage map.
2. Cluster failures (stage × object × region × clutter) into a **gap spec**, e.g. "grasp failures on guidewire coil near the table edge".
3. Choose a generator (cheapest robust first: domain_rand → mimicgen → scripted_expert → teleop) and request K episodes for the gap.
4. Run 02 on the new data, then 03 (fine-tune from the previous checkpoint on the mix), then 04.
5. Apply the promotion rule in 05.

All decisions are logged as a human-readable "learning journal". This becomes the demo narrative.

## 5. Add-on modules

### 5.1 Awareness and voice (Navneet)
- **Voice:**
  - ElevenLabs speech-to-text turns the request into text, which feeds `instruction`;
  - ElevenLabs TTS confirms ("Handing you the 6F guide catheter").
- **Proactive assist:**
  - a VLM watches the scene camera (co-human pose/hands, current procedure step);
  - a procedure-step model (a simple state machine of cathlab steps plus VLM step recognition) predicts the *next* needed tool;
  - the robot proposes the tool ("Shall I hand you the sheath?") and acts on confirmation.
- **Interface:** awareness only emits `instruction` strings into the same `PolicyAdapter`. Any VLA trained by the pipeline gains the feature without retraining. Later work: distil anticipation into the VLA via data that pairs human-context frames with tool choice.

### 5.2 Synthetic data and environment factory (Mitesh)
- DataEngine generators all emit LeRobot plus a manifest (catalogue in §8.1).
- **Env factory:**
  - a) **prompt → scene**: an agent turns text into `SceneSpec` and a USD/MJCF scene (RoboSimStudio agent / Isaac Lab tooling);
  - b) a manual scene builder from `SceneSpec` YAML;
  - c) **new objects**: pull from asset libraries, otherwise run image/text-to-3D, then fix scale, collision and mass, and register the result.
- **Demo:** generate a new cathlab-like table layout from one prompt, then generate data in it.

## 6. Evaluation research and benchmark design (Akanksh)
- Shortlist the VLAs by fit (bimanual support, LeRobot/openpi integration, fine-tune cost, latency) and freeze the choice by H6.
- Own the eval suite definition, difficulty tiers, statistics (CIs, number of seeds), and the benchmark table/plots.
- **Comparison matrix:** {scripted expert, ACT, SmolVLA, pi0.5 / GR00T / X-VLA} × {provided only, + synthetic, + loop-targeted} × {Isaac, RoboSimStudio}.

## 7. Team, ownership and timeline

| Owner | Scope |
|---|---|
| **Rakesh** | Architecture, contracts, repo/CI skeleton, Isaac env bring-up + `IsaacBackend`, 01 Ingest, 03 training runs, Loop Controller, RoboSimStudio sim2sim, integration, demo |
| **Sharan** | Literature review and option catalogue, 02 Dataset Intelligence reports, 05 registry / policy server / Docker / W&B wiring, pitch-deck research slides |
| **Navneet** | Awareness/proactive assist, ElevenLabs voice in/out, instruction interface |
| **Mitesh** | DataEngine (domain-rand, MimicGen, scripted GraspGen+cuRobo expert, teleop, Cosmos), env factory, new object creation |
| **Akanksh** | VLA research/selection, 04 eval harness design and metrics, VLM success judge, benchmarking and result plots |

**Milestones:**

| Time | Milestone |
|---|---|
| H0–2 | Contracts merged; provided env opens in Isaac; provided dataset loads in LeRobot; W&B project live |
| H2–10 | **v0** = SmolVLA on the provided handover episodes; scripted expert runs in sim; eval harness scores expert and v0; first synthetic batch; voice → instruction prototype |
| H10–20 | **v1** = provided + synthetic on 2–3 VLAs; benchmark table; prompt → scene demo; RoboSimStudio scene port |
| H20–28 | One full **loop iteration** → **v2**; sim2sim eval; awareness proposes tools in sim |
| H28–32 | Feature freeze; versioned rollout demo v0→v1→v2; recorded videos |
| H32–36 | Pitch, slides, README, final checkpoint upload |

**Working agreement:**
- Branch per person (`feat/<owner>-<topic>`) and small PRs to `main`.
- Contracts change only via a PR reviewed by Rakesh.
- Each package ships a `README` and a smoke test.
- A sync every 4 h (15 min).

## 8. Option catalogue (book-keeping; ✅ = chosen to implement, ○ = documented/optional)

### 8.1 Data generation
| Option | Status | Note |
|---|---|---|
| Domain randomization (poses, clutter, lighting, distractors, camera jitter) | ✅ first | Fastest, robust, replays the provided controller |
| MimicGen / Isaac Lab Mimic / DexMimicGen (bimanual) re-targeting of provided demos | ✅ | Scales 500 → thousands of episodes |
| Scripted expert: GraspGen grasps + cuRobo motion | ✅ | Also our "own expert policy" for the DoD |
| Teleop in sim (keyboard/SpaceMouse/VR) | ○ | For hard edge cases only |
| NVIDIA Cosmos Transfer/Predict visual augmentation | ○ stretch | Photoreal diversity; GPU-heavy |
| GR00T-Dreams / video world-model data | ○ | Documented only |

### 8.2 Models
| Option | Status | Note |
|---|---|---|
| SmolVLA (LeRobot) | ✅ baseline | Small, fast, LeRobot-native |
| pi0 / pi0.5 (openpi, LeRobot port) | ✅ candidate | Strong generalist |
| NVIDIA GR00T N1.x | ✅ candidate | Isaac ecosystem, humanoid/bimanual |
| X-VLA, OpenVLA-OFT | ○ | Evaluate if time allows |
| ACT / Diffusion Policy | ✅ lower bound | Non-language baseline |

### 8.3 Simulators
| Option | Status | Note |
|---|---|---|
| Isaac Sim / Isaac Lab (provided USD) | ✅ primary | Same scene as the provided data |
| RoboSimStudio (Newton / MuJoCo-Warp) | ✅ sim2sim | Installed locally; agentic scene building |
| MuJoCo (MJCF) / ManiSkill / Genesis | ○ | Documented alternatives |

### 8.4 Perception / grasp / judge
- GraspGen ✅.
- Contact-GraspNet / AnyGrasp ○.
- VLM judge with Qwen-VL / Gemini class models ✅.
- Grounding DINO / SAM for object grounding ○.

## 9. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Isaac env bring-up slow / version mismatch | Fix in H0–2 with a provided-docs checklist; fall back to replaying dataset video for 01–03 while sim is fixed |
| Bimanual action space unsupported by a model | Use the adapter to mask the unused arm or restrict to a single-arm subset; record in the model card |
| Training too slow for 3 versions | SmolVLA for the loop iterations; frontier models trained once on v1 data |
| Sim2sim asset conversion (USD → RoboSimStudio) breaks physics | Port a simplified scene (table + objects + robot) first; report the gap honestly |
| Synthetic data hurts performance | 02 quality gate + promotion rule; ablation shown in the benchmark |
| Integration crunch | Contracts frozen at H2; end-to-end smoke test at H10 with dummy policy |

## 10. Demo story (3 min)
1. Voice: "Hand me the guidewire." The robot does it (v0 vs v2 side by side).
2. Dataset intelligence report shows the gaps.
3. The loop journal: the gap was found, data was generated, and success rose from v0 → v1 → v2.
4. Benchmark table: VLAs × data versions × simulators.
5. One prompt creates a new scene; the same pipeline runs in it.
6. Proactive mode: the clinician reaches for the sheath step and the robot offers the sheath before being asked.
