# Sharan — Research, Dataset Intelligence, Version & Serve

> **Update (H6.5):** see the [simulation framework spec](../specs/2026-10-22-sim-framework-design.md).
> - **Coverage map** = the variation grid: object × layout × object colour × table colour × `source_sim`. Read each episode's `config` from `meta/episodes_extra.jsonl`.
> - **Gaps** are empty cells or cells where expert/policy success < threshold. Emit them as a list of `EpisodeConfig`-style cell keys for the loop.
> - **Provided FR3 Duo data:** still run the validators and readiness report on it (embodiment-agnostic demo).

You come from IT rather than robotics, so this scope plays to that: data quality, registries, APIs, Docker and research synthesis. Your agent can explain any robotics term; ask it to.

## Agent brief (paste to your agent)

> You are helping build the CathLab Companion learning loop (spec: `docs/specs/2026-10-22-cathlab-learning-loop-design.md`; contracts in `contracts/`).
>
> My scope:
> - (1) a literature review and option catalogue for robot-learning pipelines (VLAs, synthetic data, sim2sim, self-improving loops), written to `docs/research/`;
> - (2) `dataset_intel/`: validators, near-duplicate detection, a coverage map (object × table region × clutter × instruction), and an HTML/markdown readiness report with a VLA-readiness score for any LeRobot dataset;
> - (3) `serve/`: a model registry over `ModelCard` YAMLs (+ HF Hub / W&B Artifacts), a WebSocket policy server exposing `predict(obs, instruction)` for any `PolicyAdapter`, a `rollout` CLI with versioned deploy/rollback, and a Dockerfile.
>
> Write tests first; they must run without GPU using `RandomPolicy`/`DummySim` and a tiny synthetic LeRobot-like fixture.

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| S1 | Literature review (≤2 pages) + option catalogue with citations (feeds pitch "landscape" slide). Topics: SmolVLA, pi0/pi0.5, GR00T N1.x, X-VLA, OpenVLA-OFT; MimicGen/DexMimicGen, Isaac Lab Mimic, Cosmos, GR00T-Dreams; GraspGen, cuRobo; sim2real/sim2sim; self-improving/active data collection; proactive HRI | `docs/research/literature.md` | H8 draft, H20 final |
| S2 | `dataset_intel.validate(path) -> list[Issue]`: NaN/frozen frames, action jumps, gripper never closes, object never lifted, length outliers, camera desync | `dataset_intel/validators.py` | H8 |
| S3 | `dataset_intel.coverage_report(path) -> CoverageReport` with `gaps() -> list[GapSpec]`, plus near-duplicate detection | `dataset_intel/coverage.py` | H12 |
| S4 | `python -m dataset_intel.report <dataset>` → `reports/<dataset>/readiness.md` + W&B table, including the VLA-readiness score | `dataset_intel/report.py` | H14 |
| S5 | Registry: `register(card)`, `get(name, version)`, `promote(card, rule)`, `rollback(name)`; promotion rule from `configs/serve.yaml` | `serve/registry.py` | H16 |
| S6 | Policy server (WebSocket, msgpack/numpy) + client used by sim backends; latency logging | `serve/server.py`, `serve/client.py` | H20 |
| S7 | `rollout --model cathlab-pick@v2 --sim isaac\|robosimstudio`; Dockerfile; GitHub Actions running `uv run pytest` | `serve/cli.py`, `docker/`, `.github/workflows/ci.yml` | H24 |

## Interfaces

- **Produces:**
  - `Issue(episode: int, kind: str, detail: str)`;
  - `GapSpec(object: str, region: str, clutter: str, stage: str | None, count_needed: int)`, added to `contracts/` via PR;
  - `CoverageReport`;
  - registry functions above;
  - `PolicyClient(url).predict(obs, instruction) -> np.ndarray`, which implements `PolicyAdapter` so the eval runner can use a remote model.
- **Consumes:** `DatasetManifest`, `ModelCard`, `EvalReport` from `contracts/`.

## Acceptance

- The readiness report runs on the provided 500-episode dataset and flags at least the validity categories above.
- `GapSpec`s from the coverage map are consumed by Mitesh's DataEngine and Rakesh's loop.
- `rollout` switches v1 → v2 → rollback v1 in one command each, and the eval runner works against the remote server.
