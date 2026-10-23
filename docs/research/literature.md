# Literature Review: Dataset Intelligence, Evaluation, Success Judgement, Version & Serve

*Owner: Sharan · 2026-10-23 (H21) · deliverable named in [`docs/tasks/sharan.md`](../tasks/sharan.md) and [`README.md`](README.md)*

**Scope.** This review covers the parts of the challenge that Sharan's Windows CPU machine can build:
- **02 Dataset Intelligence**;
- **04 Evaluate**;
- **05 Version & Serve**;
- **Option A** (Intelligent Success Judgement), **Option B** (Self-Improving Pipeline) and **Option D** (Skill Orchestration: latency and safety guardrails).

Model selection (SmolVLA, ACT, π0, GR00T) is in [`vla_selection.md`](vla_selection.md), and data generation engines are in [`data_engines.md`](data_engines.md). Neither is repeated here.

**Verification.** Every arXiv ID below was checked against its abstract page on 2026-10-23. Two items were searched for but not found; they are listed in §7 rather than cited.

**"Runs here" legend:**

| Mark | Meaning |
|---|---|
| ✅ CPU | Can be built on the Windows laptop with numpy, pyarrow, pandas or scipy |
| 🌐 API | Needs only a hosted VLM/LLM API call (network, no local GPU) |
| 🖥️ GPU | Needs the lerobot/CUDA box, a simulator on Linux, or model training. On this machine only the CPU-side analysis can be built |

## 0. The idea in one paragraph

The challenge wants a *loop*, not a single model ("Stage 5 feeds the next learning cycle"). The 2024–2026 literature points the same way:
1. **Diversity matters more than volume.** Generalisation scales with the number of distinct environments and objects, not with demo count [10]. So the dataset report should measure *coverage cells*, not totals.
2. **Demonstrations differ in quality and can be scored without training a policy.** DemInf does it with k-nearest-neighbour mutual information [1]; SCIZOR does it with progress and deduplication [3].
3. **Evaluation has to be statistical.** With 30–50 trials per cell, point estimates mislead. Use confidence intervals, sequential tests and pairwise comparisons [12–15].
4. **Failures should be explained, not just counted.** VLM success detectors and failure-reasoning models [20–23] turn an `EvalReport` into "what data to generate next" [6, B2].
5. **Serving chunked VLAs is a latency problem.** Asynchronous inference, chunk blending and real-time chunking [26–29] decide whether a model that passes evaluation can actually be deployed.

Our pipeline already has the right skeleton (`dataset_intel`, `EvalReport` with Wilson CIs, the failure report, registry promotion). The papers below show where to add rigour.

## 1. Stage 02: Dataset Intelligence (quality, redundancy, coverage, readiness)

| # | Paper | Year · ID | Key idea | How we use it | Runs here |
|---|---|---|---|---|---|
| 1 | **Robot Data Curation with Mutual Information Estimators** ("DemInf"), Hejna et al. | 2025 · [2502.08623](https://arxiv.org/abs/2502.08623) | Scores each demo by its contribution to the state–action mutual information, using k-NN MI estimates. This captures action *diversity* and *predictability* and separates good demos from bad ones without training a policy. | **S5 `quality` score:** a k-NN (KSG) MI contribution per episode on the raw `observation.state` and `action` vectors, skipping their VAE embedding. Rank the episodes and list the bottom decile in `report.md`. | ✅ CPU (numpy k-NN; ~500 episodes is fine) |
| 2 | **CUPID: Curating Data your Robot Loves with Influence Functions**, Agia et al. | CoRL 2025 · [2506.19121](https://arxiv.org/abs/2506.19121) | Influence functions estimate each demo's effect on closed-loop success. Matches state-of-the-art results with < 33 % of the data. | Cite as the "gold standard" that our cheap scores approximate. Run it later on the GPU box with a trained SmolVLA. | 🖥️ GPU |
| 3 | **SCIZOR: Self-Supervised Data Curation for Large-Scale Imitation Learning**, Zhang et al. | 2025 · [2505.22626](https://arxiv.org/abs/2505.22626) | Filters at the transition level: (a) a task-progress predictor flags transitions that don't make progress; (b) deduplication on state–action embeddings removes redundancy. +15.4 % with less data. | **S5:** add `near_duplicate` (cosine similarity on resampled state/action windows) and extend `idle_fraction` / `frozen` toward "no progress" segments. Directly answers the brief's "invalid, redundant" question. | ✅ CPU |
| 4 | **DataMIL: Selecting Data for Robot Imitation Learning with Datamodels**, Dass et al. | 2025 · [2505.09603](https://arxiv.org/abs/2505.09603) | Fits a surrogate (datamodel) that predicts policy performance from which training data is included, then selects data end to end. | Background for Option B's "propose next data mix". Needs many training runs. | 🖥️ GPU |
| 5 | **Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning**, Hejna et al. | 2024 · [2408.14037](https://arxiv.org/abs/2408.14037) | Distributionally robust optimisation (DRO) learns per-domain mixture weights. | **S5 `object balance`** component of readiness (already in the contract). Cite as the reason imbalance is penalised. | ✅ CPU (the diagnostic) / 🖥️ (the DRO) |
| 6 | **STRAP: Robot Sub-Trajectory Retrieval for Augmented Policy Learning**, Memmel et al. | 2024 · [2412.15182](https://arxiv.org/abs/2412.15182) | Retrieves relevant *sub-trajectories* from prior data with foundation-model features and DTW. | Option B: for a failing cell, retrieve the nearest existing sub-trajectories (DTW on state) as a cheaper alternative to generating new data. | ✅ CPU (state DTW) |
| 7 | **Behavior Retrieval: Few-Shot Imitation Learning by Querying Unlabeled Datasets**, Du et al. | RSS 2023 · [2304.08742](https://arxiv.org/abs/2304.08742) | Uses a few target demos to retrieve relevant transitions from a large unlabeled set. | Same use as [6]; a k-NN over parquet state features works as a stand-in. | ✅ CPU |
| 8 | **What Matters in Learning from Offline Human Demonstrations for Robot Manipulation** (robomimic), Mandlekar et al. | CoRL 2021 · [2108.03298](https://arxiv.org/abs/2108.03298) | Policy performance is highly sensitive to demo quality and operator skill. Mixed-quality data hurts. | Justifies the per-episode `issues` list and the `valid fraction` readiness component. | ✅ CPU |
| 9 | **Decomposing the Generalization Gap in Imitation Learning for Visual Robotic Manipulation**, Xie et al. | 2023 · [2307.03659](https://arxiv.org/abs/2307.03659) | Ranks factors of variation (object position, textures, lighting, camera, distractors) by how hard they are to generalise over. | Our coverage key `(object, layout_id, obj_color_id, table_color_id, source_sim)` follows this factor view. Use their ranking to **weight** empty cells (layout > table texture > colour) in `coverage fraction`. | ✅ CPU |
| 10 | **Data Scaling Laws in Imitation Learning for Robotic Manipulation**, Lin et al. | 2024 (ICLR 2025) · [2410.18647](https://arxiv.org/abs/2410.18647) | Generalisation follows a power law in the number of *distinct environments × objects*. Past a threshold, more demos per pair add little. | **S5 headline metric:** a histogram of demos per coverage cell. Flag cells as *saturated* (above the threshold) or *starved* (below it). It turns "is the dataset ready for VLAs?" into a measurable check. | ✅ CPU |
| 11 | **Datasheets for Datasets**, Gebru et al. | CACM 2021 · [1803.09010](https://arxiv.org/abs/1803.09010) | Standard documentation: motivation, composition, collection, gaps, intended use. | `report.md` gets a short datasheet header (source, robot_type, fps, episodes, collection sim, known gaps). | ✅ CPU |

## 2. Option B: Self-improving loop (gaps → targeted data → next training)

| # | Paper | Year · ID | Key idea | How we use it | Runs here |
|---|---|---|---|---|---|
| B1 | **A Reduction of Imitation Learning … to No-Regret Online Learning** (DAgger), Ross et al. | AISTATS 2011 · [1011.0686](https://arxiv.org/abs/1011.0686) | Collect labels on the *policy's own* state distribution to fix covariate shift. | The theory behind `next_run.json`: generate data where the policy fails, not uniformly. | ✅ (concept) |
| B2 | **MimicGen: A Data Generation System for Scalable Robot Learning**, Mandlekar et al. | CoRL 2023 · [2310.17596](https://arxiv.org/abs/2310.17596) | Builds thousands of demos from a few by re-targeting object-centric sub-trajectories. | What consumes our `next_run.json` cells (Rakesh's Generate stage). The S6 recommendation "generate K targeted episodes" is MimicGen-style targeting. | 🖥️ (sim on Linux) |
| B3 | **RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots**, Nasiriany et al. | RSS 2024 · [2406.02523](https://arxiv.org/abs/2406.02523) | Generative assets + MimicGen at scale; co-training with synthetic data. | Reference for the "widen colour palette / asset set in next spec version" recommendation. | 🖥️ |
| B4 | **Robot Learning on the Job** (Sirius), Liu et al. | RSS 2023 · [2211.08416](https://arxiv.org/abs/2211.08416) | Human-in-the-loop deployment; reweights data by intervention and trust. | Option D: when a guardrail fires, log an intervention, and future training upweights those segments. | ✅ CPU (logging/weights) |
| B5 | **RoboCat: A Self-Improving Generalist Agent for Robotic Manipulation**, Bousmalis et al. | TMLR 2023 · [2306.11706](https://arxiv.org/abs/2306.11706) | The agent generates its own data. Filtered self-data improves the next iteration. | Rollouts flow back through `dataset_intel` validators *before* they rejoin the train set. | ✅ (filter) / 🖥️ (agent) |
| B6 | **Self-Improving Embodied Foundation Models**, Ghasemipour et al. | NeurIPS 2025 · [2509.15155](https://arxiv.org/abs/2509.15155) | Adds a "steps-to-go" head during supervised fine-tuning (SFT) and then uses it as reward and success detector for autonomous practice. | Cheap proxy: a normalised time-to-end label per frame lets the failure report show *how far* a failed rollout got. | ✅ (labels) / 🖥️ (training) |
| B7 | **Steering Your Diffusion Policy with Latent Space RL** (DSRL), Wagenmaker et al. | 2025 · [2506.15799](https://arxiv.org/abs/2506.15799) | Adapts a frozen behaviour-cloning (BC) diffusion or flow policy by running RL over its *noise* input; no weight fine-tuning. | A "next training strategy" recommendation when the gap is in execution rather than in data coverage. | 🖥️ |
| B8 | **AutoRT**, Ahn et al. | 2024 · [2401.12963](https://arxiv.org/abs/2401.12963) | VLM/LLM proposes diverse tasks for a fleet and filters them for safety. | Option B: an LLM reads `report.json` + failure report and drafts the next spec delta (new objects/colours), with human approval. | 🌐 API |


## 3. Stage 04: Rigorous evaluation (statistics, promotion, sim2sim)

| # | Paper | Year · ID | Key idea | How we use it | Runs here |
|---|---|---|---|---|---|
| 12 | **Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping** (STEP), Snyder et al. | 2025 · [2503.10966](https://arxiv.org/abs/2503.10966) | A sequential test for "policy A beats B" with small samples (10–50). It stops early when the answer is clear, saving trials. | **Promotion-rule upgrade (proposal):** our rule compares the candidate's Wilson *lower bound* with the baseline's *point* rate, which ignores the baseline's uncertainty. A two-sample test (STEP, or Boschloo/Barnard as a simpler first step) is more correct and needs fewer episodes. Contract change, so ask Rakesh first. | ✅ CPU (scipy) |
| 13 | **Deep RL at the Edge of the Statistical Precipice** (rliable), Agarwal et al. | NeurIPS 2021 · [2108.13264](https://arxiv.org/abs/2108.13264) | Point estimates over few runs mislead. Use stratified-bootstrap CIs, IQM and performance profiles. | Failure report: stratified bootstrap CI for the *aggregate* success across tiers and objects, next to the per-cell Wilson CIs. | ✅ CPU |
| 14 | **Robot Learning as an Empirical Science: Best Practices for Policy Evaluation**, Kress-Gazit et al. (TRI) | 2024 · [2409.09491](https://arxiv.org/abs/2409.09491) | Fix the protocol in advance, size trials for statistical power, use failure taxonomies, report reproducibly. | Justifies `min_episodes: 30`. Freeze `eval_configs(spec, tier, seeds)` as the pre-registered protocol and record it in the ModelCard. | ✅ (docs) |
| 15 | **A Careful Examination of Large Behavior Models for Multitask Dexterous Manipulation**, TRI LBM Team | 2025 · [2507.05331](https://arxiv.org/abs/2507.05331) | Large blind A/B evaluation with Bayesian/CI reporting, in sim and real. | Template for the report layout: per-condition CIs and ablations by engine, tier and object. | ✅ (docs) |
| 16 | **SIMPLER: Evaluating Real-World Robot Manipulation Policies in Simulation**, Li et al. | CoRL 2024 · [2405.05941](https://arxiv.org/abs/2405.05941) | A sim evaluation is trustworthy only if its *rankings* match real ones. Reduce control and visual gaps. | **S6 sim2sim gap:** besides the per-tier rate difference, report the **rank correlation** of versions across engines. If robosuite and MuJoCo disagree on the ranking, don't trust promotion. | ✅ CPU |
| 17 | **AutoEval**, Zhou et al. | 2025 · [2503.24278](https://arxiv.org/abs/2503.24278) | Autonomous 24/7 evaluation with learned success detectors and resets. | Architecture match for the Eval stage job queue. | 🖥️ (robot) |
| 18 | **RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies**, Atreya et al. | 2025 · [2506.18123](https://arxiv.org/abs/2506.18123) | Double-blind *pairwise* comparisons across sites aggregated into a global ranking. | Option C (Learning Network): rank versions pairwise across sites and engines instead of comparing raw rates. | ✅ CPU (ranking math) |
| 19 | **WorldEval** / **dWorldEval**, Li et al. | 2025 · [2505.19017](https://arxiv.org/abs/2505.19017) · 2026 · [2604.22152](https://arxiv.org/abs/2604.22152) | Video world models as policy evaluators. dWorldEval adds a progress token for automatic success detection. | Frontier "future work" slide. Our analogue is the cheap sim2sim screen. | 🖥️ GPU |

## 4. Option A: Intelligent success judgement and failure explanation

| # | Paper | Year · ID | Key idea | How we use it | Runs here |
|---|---|---|---|---|---|
| 20 | **Vision-Language Models as Success Detectors** (SuccessVQA), Du et al. | 2023 · [2303.07280](https://arxiv.org/abs/2303.07280) | Success detection posed as VQA over rollout frames. | Fill `judge_verdict` with 3–5 keyframes and a yes/no question; cross-check the scripted success flag and count disagreements in the failure report. | 🌐 API |
| 21 | **AHA: A VLM for Detecting and Reasoning Over Failures in Robotic Manipulation**, Duan et al. | 2024 · [2410.00371](https://arxiv.org/abs/2410.00371) | A VLM trained on procedurally perturbed failures (FailGen) explains *why* a rollout failed. | Natural-language "why" next to each top-N failing cell in `report.md`. | 🌐 API (or 🖥️ local AHA) |
| 22 | **REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction**, Liu et al. | CoRL 2023 · [2306.15724](https://arxiv.org/abs/2306.15724) | Hierarchical summary (state trace → key events → text), then an LLM explains and proposes a correction. | Send the LLM a compact **text** trace (stage timings, gripper events, object height) rather than raw video. It is cheaper, and the trace is already in `EpisodeResult`. | 🌐 API |
| 23 | **RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Correction**, Lu et al. | 2025 · [2505.12224](https://arxiv.org/abs/2505.12224) | A large failure taxonomy and QA dataset; a small model beats GPT-4o on failure analysis. | Extend our stage histogram with sub-types (e.g. grasp *miss* vs *slip*) if `EpisodeResult` carries enough signal. | ✅ (taxonomy) |
| 24 | **Vision Language Models are In-Context Value Learners** (GVL), Ma et al. | 2024 · [2411.04549](https://arxiv.org/abs/2411.04549) | Asks a VLM to order *shuffled* frames to get zero-shot task progress. Works for success detection and dataset filtering. | (a) A partial-credit progress curve per failed rollout; (b) S5 flags demos whose progress is not monotonic. | 🌐 API |
| 25 | **Unpacking Failure Modes of Generative Policies: Runtime Monitoring of Consistency and Progress** (Sentinel), Agia et al. | CoRL 2024 · [2410.04640](https://arxiv.org/abs/2410.04640) | Detects failures online from (i) inconsistency between successive action chunks and (ii) VLM progress checks. | Option D guardrail. *Offline* on this machine: compute chunk-to-chunk consistency from logged action chunks in the failure report. Online use needs the policy server on the GPU box. | ✅ (offline) / 🖥️ (online) |

## 5. Stage 05: Version & Serve (chunked VLA serving, latency, registry)

| # | Paper / source | Year · ID | Key idea | How we use it | Runs here |
|---|---|---|---|---|---|
| 26 | **SmolVLA**, Shukor et al. (Hugging Face) | 2025 · [2506.01844](https://arxiv.org/abs/2506.01844) | A small VLA plus **asynchronous inference**: execution of the current chunk overlaps with prediction of the next. | Our server/client is synchronous request/response (contract §3.2). Async (client-side queue + re-query threshold) is the next step, proposed after S2. | ✅ (protocol) |
| 27 | **LeRobot async inference** (PolicyServer / RobotClient) | docs · [huggingface.co/docs/lerobot/async](https://huggingface.co/docs/lerobot/en/async) | Reference implementation: `actions_per_chunk`, `chunk_size_threshold`, `aggregate_fn_name` for blending overlapping chunks. | Concrete parameters to mirror if we add async mode to `serve/`. | ✅ (protocol) |
| 28 | **ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware**, Zhao et al. | RSS 2023 · [2304.13705](https://arxiv.org/abs/2304.13705) | Action chunking plus **temporal ensembling** (exponentially weighted overlap of chunks). | The simplest client-side blending; pure numpy in `PolicyClient` or the executor. | ✅ CPU |
| 29 | **Real-Time Execution of Action Chunking Flow Policies** (RTC), Black et al. | NeurIPS 2025 · [2506.07339](https://arxiv.org/abs/2506.07339) | Freezes the actions already committed and inpaints the rest during flow sampling, so execution stays smooth under latency. Training-free. | Tolerance to latency changes what the `p95` budget means. Needs access to the model's denoising loop, so it lives server-side on the GPU box. | 🖥️ GPU |
| 30 | **Training-Time Action Conditioning for Efficient Real-Time Chunking**, Black et al. | 2025 · [2512.05964](https://arxiv.org/abs/2512.05964) | Moves RTC into training (simulated delay) for near-zero inference overhead. | Recommendation for the training session's next run. | 🖥️ GPU |
| 31 | **Bidirectional Decoding** (BID), Liu et al. | 2024 · [2408.17355](https://arxiv.org/abs/2408.17355) | Samples several chunks at test time and picks the one balancing coherence with the past against future quality. | An alternative to RTC; costs extra inference, so check it against the latency budget. | 🖥️ GPU |
| 32 | **FAST: Efficient Action Tokenization for VLA Models**, Pertsch et al. | 2025 · [2501.09747](https://arxiv.org/abs/2501.09747) | DCT + BPE action tokens for autoregressive VLAs (π0-FAST). | Relevant only if an autoregressive VLA is served. Background for `vla_selection.md`. | ✅ (tokenizer) |
| 33 | **openpi** (Physical Intelligence) | repo · [github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | WebSocket policy server/client for π0-family models. | Cross-check for our `/act` wire format. A π0 adapter can sit behind the same `PolicyClient`. | 🖥️ GPU (model) |
| 34 | **Hidden Technical Debt in Machine Learning Systems**, Sculley et al. | NeurIPS 2015 · [proceedings](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) | Undeclared consumers, configuration debt, hidden feedback loops. | Why the registry stores eval report paths, the promotion decision and its history, and why `/health` reports `model_version`. | ✅ |
| 35 | **Model Cards for Model Reporting**, Mitchell et al. | FAT* 2019 · [1810.03993](https://arxiv.org/abs/1810.03993) | Per-model documentation: intended use, per-condition performance, caveats. | `contracts.ModelCard` plus a link to the S5 datasheet and the S6 failure report = full provenance. | ✅ |
| 36 | **Open X-Embodiment: Robotic Learning Datasets and RT-X Models**, OXE Collaboration | ICRA 2024 · [2310.08864](https://arxiv.org/abs/2310.08864) | Pooled cross-embodiment data transfers positively. | Option C motivation: sharing validated versions across Panda ↔ FR3 Duo through one registry. | ✅ (concept) |
| 37 | **Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision**, Hoque et al. | CoRL 2022 · [2206.14349](https://arxiv.org/abs/2206.14349) | Allocates scarce human help across a robot fleet ("Return on Human Effort"). | Option C/D: route guardrail triggers to a human queue and track interventions per promoted version. | ✅ (concept) |

## 6. What to build on this machine (ranked by value ÷ effort)

None of these needs a GPU, a simulator or a contract change unless marked.

| Rank | Build | Package | Grounded in | Effort |
|---|---|---|---|---|
| 1 | **Coverage saturation / starvation** per cell (demos-per-cell histogram, factor-weighted empty cells) | S5 `coverage.py` | [10] Data Scaling Laws, [9] Decomposing | S |
| 2 | **Per-episode quality score** (k-NN mutual information on state/action) + **near-duplicate** detection | S5 `validators.py` (new kinds `low_quality`, `near_duplicate`) | [1] DemInf, [3] SCIZOR | M |
| 3 | **Stratified-bootstrap aggregate CI** and **cross-engine rank agreement** in the failure report | S6 | [13] rliable, [16] SIMPLER | S |
| 4 | **LLM/VLM failure explanation** from a text state trace (+ optional keyframes), written next to the top-N cells | S6 (optional `--judge`) | [22] REFLECT, [20] SuccessVQA, [21] AHA | M, 🌐 API key |
| 5 | **Two-sample promotion test** (Boschloo now, STEP later) as a *proposal*. Keep the current rule until Rakesh agrees | S3 follow-up | [12] STEP, [14] Kress-Gazit | S, contract change |
| 6 | **Datasheet header** in `report.md` + links from the ModelCard | S5 / S3 | [11], [35] | S |
| 7 | **Temporal-ensembling blend** in the client / executor (async mode design note) | S2 follow-up | [28] ACT, [26–27] SmolVLA async | S, contract change |

**Hand off to the GPU/Linux box** (not on this Windows PC): CUPID influence scores [2], RTC/BID server-side [29–31], online Sentinel [25], world-model evaluation [19], MimicGen generation of the `next_run.json` cells [B2], DSRL [B7].

## 7. Could not verify (do not cite)

- **"Quality over Quantity" robot-demo scoring paper:** no matching arXiv entry found. CUPID [2] and DemInf [1] cover the same ground.
- **LeRobot itself** is cited as software (repo and docs), not as a paper.
