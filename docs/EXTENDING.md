# Extending the CathLab Companion platform

This is the contract every extension (engine, object, data method, metric, report section, or anything else) is
built against: one set of stage gates, one Control Station pipeline, one deterministic Agent Brief. Read
[AGENTS.md](../AGENTS.md) first.

## 1. How the platform is put together

A **catalog item** (`configs/catalog.yaml`) becomes a **run** (`var/runs/<id>/run.yaml`, a selection of engine,
embodiment, objects, expert, variation spec and model). The Control Station **PIPELINE** tab walks the run through
**stages** (`orchestrator/stages.py`), each dispatched to a worker that executes the engine's CLI subcommand. Stage
outputs land under `var/runs/<id>/<stage>/<job>/` and become the input of the next stage.

```
catalog.yaml --[pick]--> run.yaml --[orchestrator.stages]--> engine CLI on a worker --> var/runs/<id>/<stage>/<job>/
     |                                        |                                                |
  configs/                           preview, check-expert,                         grid.png, expert_check.json,
  {catalog,workers}.yaml             generate, convert, eval,                       raw/, lerobot/, eval_report.json,
                                      rollout                                       rollout.mp4, ...
```

## 2. Stage gates

Every extension point is one **gate** in `harness/gates/<id>.yaml`: a contract, a conformance command, the paths an
agent may touch, a reference implementation and context to read first. `python -m harness brief` composes gates into
a brief; `AGENTS.md`, templates and this guide all point back to them.

| Gate | Title | Status | Owner | Extension point | Conformance |
|---|---|---|---|---|---|
| [`engine`](../harness/gates/engine.yaml) | Simulation engine plugin | ready | Rakesh (SDK, robosuite), Mitesh (ManiSkill3), Akanksh (Genesis) | `engines/<id>/` (`engine.yaml` + `cli.py` + a `SimBackend` + an `ExpertSim` adapter) | `python -m pytest -q tests/engines/test_conformance.py -k <id>` (in the engine's env) |
| [`scene`](../harness/gates/scene.yaml) | Scene and variation axes | ready | Rakesh | A new versioned spec `configs/variation/cathlab_v<N>.yaml` + `dataengine/variation.py` axes + `sim/<engine>/randomize.py` | `uv run pytest -q tests/dataengine && MUJOCO_GL=egl conda run -n robosuite python scripts/robosuite_expert_check.py --n 20` |
| [`objects`](../harness/gates/objects.yaml) | Objects on the table | partial | Rakesh | A new `VariationSpec` version, or mesh/Objaverse packages under `assets/objects/<name>/` + `assets/importers/` | `MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli check-expert --out var/check --n 20 --objects <name>` (rate >= 0.8) |
| [`expert`](../harness/gates/expert.yaml) | Expert controller | ready | Rakesh | `dataengine/expert.py` + an optional grasp provider in `dataengine/experts/<id>.py` | `MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli check-expert --out var/check --n 20` (every rate >= 0.8) |
| [`data_method`](../harness/gates/data_method.yaml) | Synthetic data method | partial | Rakesh (scripted), team (MimicGen, Cosmos, teleop) | `dataengine/registry.py` (`@register_method`) + `dataengine/<method>.py`, or an engine's `generate` stage | `uv run pytest -q tests/dataengine` |
| [`convert`](../harness/gates/convert.yaml) | LeRobot conversion | ready | Rakesh | `scripts/npz_to_lerobot.py` (panda_libero), or an engine's `convert` stage | `uv run pytest -q tests/dataengine/test_npz_sink.py tests/dataengine/test_lerobot_sink.py` |
| [`intel`](../harness/gates/intel.yaml) | Dataset intelligence | planned | Sharan | `dataset_intel/` (validators, coverage, readiness score) | `uv run pytest -q tests/dataset_intel` |
| [`train`](../harness/gates/train.yaml) | Policy training | ready | Rakesh | `train/recipes/<model>.yaml` + a `PolicyAdapter` | `uv run pytest -q tests/test_contracts.py` |
| [`eval`](../harness/gates/eval.yaml) | Evaluation | partial | Rakesh (engine eval stages), Akanksh (metrics, VLM judge) | `eval/` (metrics, judge, failure analysis), or an engine's `eval` stage | `uv run pytest -q tests/test_metrics.py tests/test_runner.py tests/test_failures.py tests/engines` |
| [`rollout`](../harness/gates/rollout.yaml) | Serving, registry and rollout | ready | Rakesh | `serve/` (registry: versions, promotion, rollback; policy server + `PolicyClient`), or an engine's `rollout` stage | `uv run pytest -q tests/serve tests/orchestrator/test_loop.py tests/engines` |
| [`assist`](../harness/gates/assist.yaml) | Awareness assistant (ASSIST) | partial | Navneet (awareness), Rakesh (panel) | `awareness/` (procedure model, VLM step tracking, `Assistant`) -> `--instruction` on a rollout | `uv run pytest -q tests/test_awareness_assistant.py tests/test_awareness_engine.py tests/test_awareness_procedure.py` |
| [`report`](../harness/gates/report.yaml) | Failure report and next run | ready | Rakesh | `report/failure_report.py` (build_report sections + next_run), `dataengine.variation.select_configs` `cells:` split | `uv run pytest -q tests/report tests/dataengine/test_targeted.py` |

## 3. Add an engine, step by step

Mirror an existing plugin: `engines/robosuite/` (MuJoCo 3.3), `engines/maniskill/` (ManiSkill3 / PhysX 5) or `engines/genesis/` (Genesis 1.4.5, own solver, CPU or GPU) — all
`panda_libero` I/O, all pass `tests/engines/test_conformance.py`.

1. Copy `engines/robosuite/` (or `engines/maniskill/`) to `engines/<id>/`.
2. Edit `engines/<id>/engine.yaml` (honest capabilities: `ready | in_progress | planned` per stage; nothing is
   `ready` until its check passes).
3. Implement the six stage functions (`preview`, `check-expert`, `generate`, `convert`, `eval`, `rollout`) with
   `engines.cli_base` (`Flag`, `Stage`, `run_cli`), following the contract:
   - `python -m engines.<id>.cli {preview|check-expert|generate|convert|eval|rollout} --out DIR [--run run.yaml]`;
   - exit 0 on success, non-zero with a **one-line reason on stderr**;
   - progress on stdout;
   - outputs only under `--out`;
   - simulator imports are **lazy** (inside stage functions), so `python -m engines.<id>.cli --help` works in the
     core env and in the `lerobot` env;
   - `--run <run.yaml>` restricts objects through `selection.objects`.
4. Add an `ExpertSim` (`dataengine.expert.ExpertSim`) and a `SimBackend` in `sim/<id>/`, so
   `dataengine.expert.run_expert_episode` and `engines.expert_policy.ExpertPolicy` run unchanged. Core frame: robot
   base, table top z = 0, quaternions wxyz. The same `EpisodeConfig` seed is the same episode in every engine.
5. Run the CLI dry-run tests in the core env, then the conformance test in the engine's env:
   `python -m pytest -q tests/engines/test_conformance.py -k <id>`.
6. Add a worker entry in `configs/workers.yaml` and start it:
   `python -m orchestrator.worker --name <id> --capabilities scene,generate,eval --port 87xx`.
7. Add the catalog item (`configs/catalog.yaml`) and a `CREDITS.md` row for the simulator, and record what passed
   in the PR description.

## 4. Add an object, data method, metric or report section

- **Object:** add a primitive object to a new versioned `configs/variation/cathlab_v<N>.yaml` (never edit
  `cathlab_v1.yaml`), with a grasp the analytic expert understands (`dataengine.expert.grasp_pose`). See the
  [`objects`](../harness/gates/objects.yaml) gate and the `add_object` harness template.
- **Data method:** register a new synthetic data generator with `@register_method` in `dataengine/registry.py` and
  implement it in `dataengine/<method>.py`, writing raw shards in the same manifest format as scripted generation.
  See the [`data_method`](../harness/gates/data_method.yaml) gate and the `add_data_method` template.
- **Metric:** add a metric reading `EvalReport` JSON only (so it works for every engine) under `eval/`. See the
  [`eval`](../harness/gates/eval.yaml) gate and the `add_eval_metric` template.
- **Report section:** add a section to `report/failure_report.py` that reads `EvalReport`s and contributes to
  `report.md` / `next_run.json`. See the [`report`](../harness/gates/report.yaml) gate and the `add_report_section`
  template.

## 5. Close the loop (versions → report → next run)

Once a checkpoint has an `EvalReport`, the loop from PIPELINE eval back to a new targeted generate is:

```
PIPELINE eval → MODELS register → PROMOTE → REPORT → CREATE NEXT RUN → PIPELINE generate (cells:) → convert → train → eval
```

PIPELINE also has these steps as stages: **train** (`python -m train.recipe <model> --run` on a worker advertising
`train`, recipes in `train/recipes/`), **register** (the run's latest eval report + the checkpoint → next
`cathlab-pick@vN`) and **promote** (the same rule as below), the last two executed by the control plane itself.

- `python -m serve.registry register --card <ModelCard.yaml> --eval robosuite=<eval_report.json> [--eval maniskill=...]`
  adds a version (`cathlab-pick@vN`) under `var/registry/`; `promote <name> <version>` applies the rule in
  `configs/serve.yaml` (CI low on the primary engine (robosuite) > production success rate, ≥ `min_episodes`,
  p95 latency ≤ `latency_budget_ms_p95`; `--force` only skips the CI-beats-production check, never latency);
  `rollback <name>` restores the previously promoted version. The MODELS tab calls the same `orchestrator/loop.py`
  API (register, promote, rollback).
- `python -m serve.policy_server --registry cathlab-pick@production` (or `make policy-server`) serves the production
  version over HTTP+JSON on `:8720` (token auth, `X-Cathlab-Token`); `serve/policy_client.py` is the stdlib+numpy
  client every engine's `--policy url:<server>` uses.
- `python -m report.failure_report --eval robosuite=<eval_report.json> [--eval maniskill=...] --model cathlab-pick@vN --out var/reports/<id>`
  builds `report.md` / `report.json` (failing cells, sim2sim gap, stage histogram, recommendations) and
  `next_run.json`, which lists only **train-split** cells (`layout_id`, `obj_color_id`, `table_color_id` all in the
  spec's train splits) so a targeted run never trains on held-out eval cells. The REPORT tab's **CREATE NEXT RUN**
  button writes a run selection with `split: cells:<next_run.json>` that PIPELINE's **generate** stage prefills.
- Policy strings every engine CLI accepts: `expert`, `random`, `lerobot:<hf-repo-or-path>`, `url:<policy-server-url>`.
  The control plane (and only the control plane) also resolves `registry:<name>@<version|production>` to a policy
  string before dispatch (`orchestrator/pipeline.py:Pipeline.effective_params`, via `serve.registry.Registry`).
- **Add a report section:** add a function to `report/failure_report.py` that reads `EvalReport`s and contributes to
  `build_report`'s sections (feeding `report.md` and, if it names cells, `next_run.json`). See the
  [`report`](../harness/gates/report.yaml) gate and the `add_report_section` harness template.

## 6. Agent Brief

A deterministic markdown brief for any coding agent, built from a plain-English request plus the stage gates it
touches:

- **UI:** the **AGENT BRIEF** button (every tab), or **AGENT BRIEF FOR THIS STAGE** in **PIPELINE**.
- **CLI:** `python -m harness brief "Add a MuJoCo engine" --stage engine --engine mujoco --out brief.md`.
- **API:** `POST /api/agent/brief`.

Copy the brief into any coding agent (GitHub Copilot, Claude Code, Codex, Cursor), or push it straight into the
harness with **RUN WITH AGENT**.

## 7. Rules

Same seven lines as `harness/brief.py`'s `RULES`:

- Read AGENTS.md first and follow it.
- Branch from the tip of origin/main; small Conventional Commits.
- Commits are authored by the human only: no Co-authored-by trailers for AI agents.
- Never commit data, checkpoints, .env or secrets; outputs go under var/ or data/ (git-ignored).
- Never edit configs/variation/cathlab_v1.yaml (the published v1 dataset depends on it); add a new versioned spec.
- Add a row to CREDITS.md for every project, model, dataset or paper you use.
- Record decisions and evidence (check results) in the PR description.
