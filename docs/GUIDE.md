# CathLab Control Station: user guide

## What it is

The Control Station is the UI + agent harness of the cathlab learning loop: a simulation-first pipeline that
generates data, trains and evaluates policies, registers and promotes model versions, and hands instruments to a
clinician in simulation. It is one FastAPI control plane (`orchestrator/control.py`) serving a static frontend
(`frontend/`) plus one or more workers (`orchestrator/worker.py`) that actually run the engine CLIs. See
[README.md](../README.md) for the architecture diagram and [DEMO.md](DEMO.md) for the 5-minute walk-through.

## First 15 minutes

```bash
cd vla-cathlab-companion && git pull --rebase && uv sync
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost   # proxy machines: else 407 on every localhost call
make demo-bootstrap                 # var/registry + production failure report from the committed eval reports
make dev                             # terminal 1: control plane + UI on http://localhost:8700
```

Start one worker per terminal (env = the eval env with torch + lerobot + robosuite, e.g. `~/envs/vla-eval`):

```bash
# terminal 2: robosuite (scene, generate, eval, rollout)
MUJOCO_GL=egl ~/envs/vla-eval/bin/python -m orchestrator.worker --name robosuite --capabilities scene,generate,eval --port 8711
# terminal 3: lerobot (convert, intel, train). Do NOT run a full training on an 8 GB laptop.
~/envs/vla-eval/bin/python -m orchestrator.worker --name lerobot --capabilities convert,intel,train,policy_server --port 8712
# optional terminal 4: Genesis (only if ~/venvs/genesis exists here)
CATHLAB_GENESIS_BACKEND=cpu PYTHONPATH=. ~/venvs/genesis/bin/python -m orchestrator.worker --name genesis --capabilities scene,generate,eval --port 8715
```

- `pip install fastapi uvicorn httpx` once in the worker env, if `python -m orchestrator.worker` fails to import.
- **ManiSkill can't run on WSL2** (no Vulkan): its lamp stays red and its stages say `worker 'maniskill' is offline`.
  That is an expected result, not a bug.
- Reset the demo state at any time with `make demo-reset` (or re-run `make demo-bootstrap`).

## Screen layout

**Header:** brand `CATHLAB CS`, 8 tabs (PROJECT, PIPELINE, DATA, MODELS, REPORT, ASSIST, AGENT, LOG), the
**AGENT BRIEF** button (opens the brief drawer, any tab), and worker lamps (green = healthy, red = offline; hover
shows the URL and capabilities, or the error).

**Footer:** **▶ RUN STAGE** (runs the selected PIPELINE stage; disabled unless a stage is selected and enabled),
**■ STOP** (cancels the running job of the selected stage; disabled unless a stage is running), and a status line
(`#status`) that shows the last footer message, e.g. `READY`, `RUN <id> CREATED`, `STARTED SCENE PREVIEW (<job id>)`,
`STAGE REJECTED: <reason>`.

## PROJECT

Build a run: pick one catalog item per section (simulator, embodiment, objects, expert, variation, model), see
what's wrong with the selection, and create the run.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| Tree item (e.g. `simulator`) | Switches SECTION to that catalog section | — | Section's items as cards with a status badge (`● READY` / `● IN PROGRESS` / `● PLANNED`) | — |
| Item card (click) | Selects it (single-select sections) or toggles it (multi-select, e.g. objects) | `POST /api/selection/check` | SELECTION CHECK re-renders | — |
| **Build with agent** (on a non-ready item or a SELECTION CHECK row) | Switches to AGENT with the matching template prefilled | — | AGENT tab opens with the form filled; job not started | — |
| `+ Add <section> with agent` | Opens the section's `add_task` template, unprefilled | — | AGENT tab opens | — |
| Run note (text field) | Free text attached to the run | — | — | — |
| **CREATE RUN** | Creates the run from the current selection | `POST /api/runs` | Footer `RUN <id> CREATED`; CURRENT RUN shows the id and selection; PIPELINE's RUN list refreshes | Footer `RUN REJECTED: <message>` (selection not runnable); button is disabled while SELECTION CHECK lists problems |

SELECTION CHECK shows `✓ Selection is runnable.` when there are no problems, otherwise one line per problem, e.g.
`<Section>: nothing selected`, `<item label> is <status>`, `<item label> requires <other> in [...]`,
`<Section>: select at least <min>`, `<Section>: unknown item '<id>'`.

## PIPELINE

Pick a run, see which of its 9 stages can start (and why not), run one, follow its log and outputs.

**Stage lamp states** (`frontend/js/pipeline.js`): **running** (job state `queued`/`running`), **done** (job state
`done`), **failed** (any other job state, e.g. `failed`/`cancelled`), **ready** (`enabled: true`, no job),
**blocked** (`enabled: false`, no job; hover the lamp for the reason).

**Blocking reasons** (`orchestrator/pipeline.py`):
- `next run file missing: <path>` (generate stage of a next run, when the cells file is gone)
- `no engine plugin '<id>' (engines/<id>/engine.yaml)`
- `<engine>: '<capability>' is planned` / `<engine>: '<capability>' is in progress`
- `no online worker with capability '<worker_capability>'`
- `worker '<name>' is offline`
- `needs '<stage>' to finish first`
- (register/promote, local stages) same `needs '...'` rule, keyed on `register`/`eval`

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| **RUN** select | Switches the active run | `GET /api/runs/{run_id}/stages` | Strip of 9 stage tiles, each with its lamp and sub-label (`ready`/`blocked`/job state) | If there are no runs: option "No runs yet: CREATE RUN in PROJECT" |
| Stage tile (click) | Selects it for the detail pane | — | STAGE title + detail: `BLOCKED: <reason>` or "Runs on worker <name>." / "Runs in the control plane (model registry)."; its param form | — |
| **AGENT BRIEF FOR THIS STAGE** | Opens the brief drawer with that stage's gate and the run's engine pre-ticked | — | Drawer opens | — |
| **▶ RUN STAGE** (footer) | Starts the stage with the form's params | `POST /api/runs/{run_id}/stages/{stage_id}` | Footer `STARTED <STAGE LABEL> (<job id>)`; live console; OUTPUTS once done | Footer `STAGE REJECTED: <reason>` (e.g. a required param missing: `stage '<id>' needs <names>`; or the stage became blocked) |
| **■ STOP** (footer) | Cancels the stage's running job | `POST /api/jobs/{job_id}/cancel` | Footer `STOP SENT (<job id>)`; lamp moves to failed/cancelled once the worker confirms | Footer `STOP FAILED: <reason>` (e.g. `cancel failed: <error>`) |
| Output file link | Opens/downloads a job output | `GET /api/jobs/{job_id}/file?path=...` | File opens in a new tab; images/`.mp4`/`.gif` also inline (first 4) | `no file '<rel>'`; `path is outside the job's outputs` |

### Stages (in order, `orchestrator/stages.py`)

#### Scene preview
- **Runs:** engine CLI `preview` (e.g. `python -m engines.robosuite.cli preview`)
- **Where:** the run's engine worker (`manifest.worker`)
- **Params:** `n` (default `8`) — train configs in the grid
- **Blocks:** engine plugin missing, `preview` capability not ready, engine worker offline
- **Outputs:** `grid.png` (7 train configs + 1 all-held-out)

#### Check expert
- **Runs:** engine CLI `check-expert`
- **Where:** engine worker
- **Params:** `n` (default `5`, episodes per object), `split` (default `train`)
- **Blocks:** `expert.<expert_id>` capability not ready, engine worker offline
- **Outputs:** `expert_check.json`; per-object success printed in the log

#### Generate data
- **Runs:** engine CLI `generate`
- **Where:** engine worker
- **Params:** `n` (default `30`, configs — all objects, balanced), `start` (default `0`, first config of the window), `split` (default `train`, or `cells:<next_run.json>` for a targeted next run)
- **Blocks:** `generate` capability not ready, engine worker offline, `next run file missing: <path>` (next-run generate only)
- **Outputs:** `raw/shard_*/episodes/*.npz`, `manifest.yaml`, `episodes_extra.jsonl`

#### Convert to LeRobot
- **Runs:** engine CLI `convert`
- **Where:** any healthy worker advertising capability `convert` (e.g. lerobot)
- **Params:** — (takes `generate`'s output folder as `--dataset`)
- **Blocks:** `needs 'generate' to finish first`; `no online worker with capability 'convert'`
- **Outputs:** `lerobot/` LeRobot v3 dataset

#### Evaluate
- **Runs:** engine CLI `eval`
- **Where:** engine worker
- **Params:** `policy` (default `expert`; `expert | random | lerobot:<hf-repo-or-path> | url:<policy-server-url> | registry:<name>@<version|production>`), `model-version` (label in the report), `tiers` (default `in_distribution`; comma-separated, empty = all suite tiers), `max-seeds` (default `1`; seeds per tier, empty = all)
- **Blocks:** `eval` capability not ready, engine worker offline; `registry:...` policies also need a resolvable registry entry (`cannot resolve registry:<ref>: <error>`)
- **Outputs:** `eval_report.json`, failure videos

#### Rollout
- **Runs:** engine CLI `rollout`
- **Where:** engine worker
- **Params:** `policy` (default `expert`, same syntax as Evaluate), `object` (default: first object of the run), `seed` (default `5000`), `instruction` (override)
- **Blocks:** `rollout` capability not ready, engine worker offline
- **Outputs:** `rollout.mp4`/`.gif`, `rollout.json`

#### Train policy
- **Runs:** `python -m train.recipe --run`
- **Where:** any healthy worker advertising capability `train`
- **Params:** `model` (default `smolvla`; recipe: `act | smolvla | xvla` in `train/recipes/`), `dataset` (required, `<hf-repo>@<revision>`), `repo-id` (required, checkpoint repo, e.g. `<user>/smolvla_panda_pick_v3`), `steps` (default: the recipe's), `io` (default `panda_libero`), `dataset-root` (local LeRobot folder instead of the Hub)
- **Blocks:** `no online worker with capability 'train'`
- **Outputs:** checkpoint pushed to `repo-id`

#### Register version
- **Runs:** control plane action (local; `orchestrator/loop_stages.py`), no worker dispatch
- **Where:** local (control plane)
- **Params:** `ckpt` (HF repo or path of the evaluated checkpoint), `datasets` (comma-separated `<repo>@<revision>` it was trained on)
- **Blocks:** `needs 'eval' to finish first`; missing `ckpt` (`register needs 'ckpt' (the HF repo or path that the eval stage evaluated)`); eval has no `eval_report.json`
- **Outputs:** new `cathlab-pick@vN` entry, visible in MODELS

#### Promote to production
- **Runs:** control plane action (local; the promotion rule)
- **Where:** local (control plane)
- **Params:** `version` (default: the version this run's Register stage created)
- **Blocks:** `needs 'register' to finish first`
- **Outputs:** `PROMOTED <version>: <reason>` or `NOT PROMOTED <version>: <reason>` written to the job's result/log

### Policy strings

`expert`, `random`, `lerobot:<hf-repo>`, `url:<policy-server>`, `registry:cathlab-pick@<vN|production>` (resolved to
`expert` or `lerobot:<ckpt_uri>` by the registry before the job starts).

## DATA

Published datasets (`configs/datasets.yaml`, pinned HF revisions that training uses) and local generate outputs
(every `manifest.yaml` under `var/runs/`).

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| (tab opens) | Loads both tables | `GET /api/datasets` | PUBLISHED DATASETS: dataset@revision (linked to huggingface.co, unless local), robot, engine, physics, episodes, frames, fps, visibility badge, notes. LOCAL GENERATE OUTPUTS: dataset@version, scene, source, episodes, run id, path, notes | Footer `DATASETS: <error>` |
| **REFRESH** | Re-fetches both tables | `GET /api/datasets` | Same as above | Same as above |

If there are no local outputs yet: "No generate outputs yet: run GENERATE in PIPELINE." under LOCAL GENERATE OUTPUTS.

## MODELS

The model registry: register an evaluated checkpoint as a version, promote it to production by the CI rule, roll
back, and jump to REPORT for any version's eval reports.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| EVAL REPORTS picker (one dropdown per engine) | Picks the `eval_report.json` to attach per engine | `GET /api/evals` (populates options) | Dropdown entries `<model> · <rate>% (n=<n>) · <path>` | "No eval reports yet: run EVALUATE in PIPELINE." |
| **REGISTER** | Registers a new version | `POST /api/registry/register` | Footer `REGISTERED <model_name>@<version>`; new VERSIONS row | Footer `REGISTER REJECTED: <message>` |
| **PROMOTE** (per version row) | Tries to promote that version (success check + latency check) | `POST /api/registry/promote` (`force: false`) | Footer `PROMOTED <version>: <reason>` or `NOT PROMOTED <version>: <reason>` | Footer `PROMOTE FAILED: <message>` |
| **FORCE** (per version row) | Promotes skipping the success check; latency is still checked | `POST /api/registry/promote` (`force: true`) | Same as PROMOTE; reason `forced by user` on success | Still refused on latency: `latency p95 <x> ms > budget <y> ms` |
| **REPORT** (per version row) | Jumps to REPORT with that version's model label and eval reports prefilled | — | REPORT tab opens, fields filled | — |
| **ROLLBACK** | Makes the previously promoted version production again (confirm dialog) | `POST /api/registry/rollback` | Footer `ROLLED BACK: production is <version or none>`; HISTORY shows a `rollback` row | Footer `ROLLBACK FAILED: <message>`; button disabled if there is no production version |
| **REFRESH** | Reloads evals + registry | `GET /api/registry` | VERSIONS and HISTORY reload | Footer `REGISTRY: <error>` |

PRODUCTION shows `<model_name>@<version>` or `none`. The promotion rule line reads: *"Promote if the 95% CI low on
`<primary_engine>` beats production, with ≥ `<min_episodes>` episodes and p95 latency ≤ `<latency_budget_ms_p95>` ms.
FORCE skips the success check, never the latency check."* VERSIONS columns: VERSION, MODEL (`expert` or the
`ckpt_uri`), one column per engine with success % and CI, STATE (`PRODUCTION` badge or `candidate`), actions.
Promotion rejection reasons seen in practice: `no eval on <engine>`, `insufficient eval on <engine>: <n> < <min>
episodes`, `latency p95 <x> ms > budget <y> ms`, `<version> is already production`, `does not beat production
<base_version>: CI low <x>% <= <y>%`.

## REPORT

Cross-physics summary, failure reports built from eval reports, and the targeted next run.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| (tab opens) | Loads CROSS-PHYSICS, EVAL REPORTS pickers, REPORTS list | `GET /api/cross-physics`, `GET /api/evals`, `GET /api/reports` | Table rows: model · trained on · one column per engine (`<rate>% (<ci_low>–<ci_high>%), n=<n>`, bold = in-domain) | Footer `CROSS-PHYSICS: <error>`; "No cross-physics table yet: run python -m train.cross_physics_report." |
| **REFRESH** (CROSS-PHYSICS) | Reloads the table | `GET /api/cross-physics` | Same as above | Same as above |
| Model label field | Free text, e.g. `cathlab-pick@v3` | — | — | — |
| EVAL REPORTS picker | Picks per-engine eval reports for the new report | `GET /api/evals` | Dropdown options as in MODELS | — |
| `k` (episodes per failing cell) | Number, default `8` | — | — | — |
| **BUILD REPORT** | Builds a failure report | `POST /api/reports` | Footer `REPORT <id> READY`; appears in REPORTS; REPORT pane shows the markdown | Footer `REPORT REJECTED: <message>` (e.g. "at least one eval report is required") |
| Item in REPORTS list | Opens that report | `GET /api/reports/{id}` | REPORT pane: markdown (success %, failing cells, sim2sim gap, recommendations); next-run info line | Footer `REPORT: <error>` |
| **CREATE NEXT RUN** | Creates a run targeting the report's train-split failing cells | `POST /api/reports/{id}/next-run` | Footer `NEXT RUN <id> CREATED: run GENERATE DATA in PIPELINE`; switches to PIPELINE on the new run | Footer `NEXT RUN REJECTED: <message>` (e.g. "the report has no train-split failing cells to target"); button disabled when the open report has no failing cells |

NEXT RUN info line: `Next run: <n> train-split failing cells, <k> targeted episodes (GENERATE DATA in PIPELINE).` or
`No train-split failing cells: follow the recommendations (widen the spec) instead.`

## ASSIST

A procedure step proposes the object the clinician needs; **YES** hands it over as a rollout of the chosen policy
with that instruction.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| **PROCEDURE** select | Picks the procedure | `GET /api/assist/procedures` | CURRENT STEP list (`<n>. <step_id>` with its tools, or `—`) | — |
| Step (click) | Proposes the object for that step | `POST /api/assist/propose` | PROPOSAL: `Step <id>: Shall I hand you the <object>?  (<reason>)` + instruction, or `Step <id>: <reason>.` when nothing is needed; YES/NO enable | Footer `ASSIST: <error>` |
| **run** select | Picks which run's simulator/objects to use | `GET /api/runs` | — | — |
| **policy** select | `production model (registry)` (`registry:cathlab-pick@production`) or `scripted expert` (`expert`) | — | — | — |
| **YES, HAND IT OVER** | Starts the rollout | `POST /api/assist/handover` (wraps the Rollout stage) | Footer `HANDOVER <OBJECT> STARTED (<job id>)`; console streams; video appears in OUTPUTS when done | Footer `HANDOVER REJECTED: <reason>` |
| **NO** | Clears the proposal | — | PROPOSAL resets to "Click the step the clinician is in." | — |
| **REFRESH** (RECORDED ROLLOUTS) | Reloads the video gallery | `GET /api/videos` | Cards: pre-recorded videos (`configs/demo_videos.yaml`) with a ✓/✗ success mark, plus local rollouts with object/policy/success/steps | "No recorded rollouts yet: run a handover, or list videos in configs/demo_videos.yaml." |

## AGENT

Run a task template with a coding agent in an isolated git worktree; review the diff and push the branch.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| Template (click) | Opens its form, empty or prefilled from PROJECT's "Build with agent" | `GET /api/agent/tasks` (loaded once) | Form with the template's params, extra instructions, agent select, "Allowed paths: ... · Checks: ..." | "Agent harness is not mounted on this control plane (start it with the agent enabled)." |
| Existing job (click) | Opens that job | `GET /api/agent/jobs/{id}` | PROMPT / DIFF / PUSH BRANCH / DISCARD buttons; console shows the job log | — |
| **Start agent** | Submits the form and starts the job | `POST /api/agent/jobs` | Footer `AGENT JOB <id> STARTED`; opens the job view | Footer `AGENT REJECTED: <message>` |
| **PROMPT** | Shows the generated prompt | `GET /api/agent/jobs/{id}/prompt` | Prompt text in the console pane | — |
| **DIFF** | Shows the agent's diff | `GET /api/agent/jobs/{id}/diff` | Diff text in the console pane | — |
| **PUSH BRANCH** | Pushes the job's branch | `POST /api/agent/jobs/{id}/push` | Footer `PUSHED <branch>`; enabled only when job state is `ready_for_review` | Footer `PUSH FAILED: <message>` |
| **DISCARD** | Discards the job's worktree/branch | `POST /api/agent/jobs/{id}/discard` | Footer `DISCARDED`; disabled while the job is active or already discarded/pushed | Footer `DISCARD FAILED: <message>` |

Job meta line: `STATE <state>` (+ `ERROR <message>` if any), `AGENT <name>`, `BRANCH <branch>`, `FILES <changed
files>`, `CHECKS <✓/✕ name ...>`. See [HARNESS.md](HARNESS.md) for the full job lifecycle and task template list.

## LOG

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| (tab open, auto-refresh every 4 s) | Lists every stage/handover/agent job | `GET /api/jobs`, `GET /api/agent/jobs` | Table: ID, KIND, NAME, STATE (badge), UPDATED (time), newest first | — |

## AGENT BRIEF drawer

Available from the header on every tab. Describe anything the UI doesn't offer; get a markdown brief with the
relevant stage contracts, tests, templates and rules for any coding agent.

| Control | What it does | API | What you see | Common errors |
|---|---|---|---|---|
| **AGENT BRIEF** (header) | Opens the drawer (optionally pre-ticks stages/engines passed by the caller, e.g. PIPELINE's "AGENT BRIEF FOR THIS STAGE") | — | Drawer with request textarea, STAGES checks, ENGINES checks | — |
| Request textarea | Free text, e.g. "Add a MuJoCo engine" | — | — | — |
| STAGES checks | Tick the gates (stage contracts) the brief should include | `GET /api/agent/gates` (loaded once) | — | — |
| ENGINES checks + "new engine id" field | Tick existing engines and/or add a new engine id | `GET /api/engines` (loaded once) | — | — |
| **BUILD BRIEF** | Builds the markdown brief | `POST /api/agent/brief` | Footer `BRIEF READY`; markdown appears in the drawer; COPY/DOWNLOAD/RUN enable | Footer `BRIEF REJECTED: <message>` |
| **COPY** | Copies the brief to the clipboard | — (clipboard API) | Footer `BRIEF COPIED` | — |
| **DOWNLOAD .MD** | Downloads `agent-brief.md` | — | File download | — |
| **RUN WITH AGENT** | Closes the drawer, switches to AGENT, opens the `free_task` template prefilled with the brief as the goal and `uv run pytest -q` as the check | — | AGENT tab opens, form filled, job not started | — |
| **CLOSE** | Closes the drawer without running anything | — | Drawer hides | — |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Every call fails, lamps red, footer 407 | HTTP proxy catches localhost | `export NO_PROXY=127.0.0.1,localhost` before `make dev` and the workers |
| Stage `worker '<x>' is offline` | worker not started / wrong port | start it (First 15 minutes); ports 8711 robosuite, 8712 lerobot, 8714 maniskill, 8715 genesis |
| ManiSkill stages always offline here | WSL2 has no Vulkan | run ManiSkill on a GPU machine; expected |
| Production is not v2 | leftovers from earlier tests | `make demo-reset` |
| First rollout slow | HF checkpoint download | run one rollout before the demo (cached afterwards) |
| Live rollout too slow for the demo | 8 GB laptop | ASSIST → RECORDED ROLLOUTS, or policy = scripted expert |
| Footer `STAGE REJECTED: stage '<id>' needs <names>` | a required param (e.g. Train's `dataset`/`repo-id`) was left empty | fill it in the stage form before RUN STAGE |
| Footer `STAGE REJECTED: no online worker with capability '<cap>'` | no worker advertises that capability | start a worker with `--capabilities <cap>,...` |
| Footer `REGISTER REJECTED: ...` | no eval report ticked, or missing `ckpt_uri` for a lerobot adapter | tick at least one eval report; fill the checkpoint field |
| Footer `NOT PROMOTED <v>: does not beat production <base>: CI low <x>% <= <y>%` | candidate's CI low doesn't beat production's rate | use FORCE (still latency-checked), or collect more episodes |
| Footer `PROMOTE FAILED: ...` with `latency p95 <x> ms > budget <y> ms` | candidate is too slow | FORCE does not help here: latency is always checked |
| "Agent harness is not mounted on this control plane" (AGENT tab) | control plane started with `--no-agent`, or the harness failed to init | start without `--no-agent`; check `make dev`'s logs |

## Known issues

| Issue | Where | Workaround |
|---|---|---|
| ManiSkill stages always say `worker 'maniskill' is offline` on the RTX 2000 Ada | WSL2 has no Vulkan | Run ManiSkill on the RTX 6000 Ada; expected here |
| Genesis worker not started on the RTX 2000 Ada (no `~/venvs/genesis`) | setup | Its lamp stays red; Genesis stages blocked. Not needed for the demo |
| `rollout.mp4` from SmolVLA doesn't play in some Chromium builds without H.264 | browser codecs | Use regular Chrome/Edge; GIF rollouts (expert) play everywhere |
| After `git pull`, the browser may keep old JS modules | browser cache | Hard reload (Ctrl+Shift+R) |
| PROMOTE stage shows lamp **failed** when the rule says NOT PROMOTED | PIPELINE | Expected: the reason is in the log (e.g. `insufficient eval on robosuite: 3 < 30 episodes`) |
| A 409 on CREATE RUN shows only the first problem message in the footer | PROJECT | The SELECTION CHECK list shows every problem |
| SmolVLA live rollouts take ~2–4 min on the 8 GB laptop (~444 ms per predict) and the syringe can fail at grasp | ASSIST / PIPELINE | Demo the **vial** (succeeded in 125 and 149 steps today); RECORDED ROLLOUTS or the scripted expert as fallback |
| DOWNLOAD .MD in the AGENT BRIEF could not be confirmed by browser automation (blob link) | AGENT BRIEF | Check by hand once |

Fixed during the click test (commit `71649b1`): AGENT BRIEF FOR THIS STAGE on REGISTER/PROMOTE now ticks the `rollout` gate
(was `report`); OUTPUTS for control-plane stages no longer logs a 404; X-VLA listed as a policy model, π0.5 marked dropped.

## Click test results

Run on the RTX 2000 Ada (WSL2, 8 GB), 2026-10-24 07:28–07:42, `main` at `0986dfb` + fixes. Workers: robosuite + lerobot (`~/envs/vla-eval`); ManiSkill and Genesis not started.

| # | Step | Result | Note |
|---|---|---|---|
| A1 | Open http://localhost:8700 | PASS | `CATHLAB CS`, 8 tabs, AGENT BRIEF, footer `READY`, ▶ RUN STAGE / ■ STOP disabled |
| A2 | Hover each worker lamp | PASS | robosuite + lerobot green with capabilities; maniskill + genesis red (`Connection refused`, not started here) |
| B1 | Click each node in PROJECT TREE | PASS | 6 sections with READY / IN PROGRESS / PLANNED badges |
| B2 | robosuite / panda / 3 objects / analytic / smolvla | PASS | `✓ Selection is runnable.` |
| B3 | Select planned simulator `newton` | PASS | `Newton 1.6 (Warp, GPU) is planned` + BUILD WITH AGENT; also `Franka Panda (single arm) requires simulator in [...]`; CREATE RUN disabled |
| B4 | Build with agent on Newton | PASS | AGENT tab, template `Add a simulation engine plugin`, `engine_id=newton` |
| B5 | Re-select robosuite, note, CREATE RUN | PASS | `RUN run-1024072904-0ad5 CREATED`; CURRENT RUN shows the selection |
| C1 | Open PIPELINE, pick the run | PASS | 9 stages; convert `needs 'generate'`, register `needs 'eval'`, promote `needs 'register'`; TRAIN ready (lerobot worker has `train`) |
| C2 | Click a blocked stage | PASS | `BLOCKED: needs 'generate' to finish first`; RUN STAGE disabled |
| C3 | Scene preview n=4 | PASS | `STARTED SCENE PREVIEW`; done; OUTPUTS `grid.png` thumbnail |
| C4 | Check expert n=2 | PASS | bottle/vial/syringe 2/2 (100 %), `expert_check.json` |
| C5 | Generate data n=3 | PASS | `kept 3/3 (100%)`; 3 `.npz`, `manifest.yaml`, `episodes_extra.jsonl`; CONVERT becomes ready |
| C6 | Convert to LeRobot | PASS | on the lerobot worker; `wrote 3 episodes`; `lerobot/` files in OUTPUTS; listed in DATA |
| C7 | Evaluate expert, in_distribution, 1 seed | PASS | 3/3 OK, `success 100% (95% CI 44%-100%)`; REGISTER becomes ready |
| C8 | Rollout `registry:cathlab-pick@production`, vial, seed 5001 | PASS | SmolVLA from HF: `success in 125 steps`; `rollout.mp4` in OUTPUTS |
| C9 | ■ STOP during Check expert | PASS | `STOP SENT (…)`; job CANCELLED, lamp failed |
| C10 | Register version | PASS | `registered cathlab-pick@v8 … with the robosuite eval` (ckpt/datasets params optional) |
| C11 | Promote to production | PASS (expected refusal) | log `insufficient eval on robosuite: 3 < 30 episodes`; lamp failed |
| C12 | AGENT BRIEF FOR THIS STAGE | PASS after fix | opened with the stage's gate + engine `robosuite`; REGISTER/PROMOTE ticked `report` → fixed to `rollout` (`71649b1`) |
| C13 | Train policy (inspect only) | PASS | params model=smolvla, dataset, repo-id, steps, io, dataset-root; runs on worker lerobot (not started) |
| D1 | Open DATA | PASS | 5 published datasets with HF links (pick_robosuite, pick_maniskill, pick_genesis, pick_panda_v2, FR3 Newton) |
| D2 | REFRESH | PASS | LOCAL GENERATE OUTPUTS lists the C5 generate and C6 convert outputs (3 episodes) |
| E1 | Open MODELS | PASS | PRODUCTION `cathlab-pick@v2`; rule sentence; v1…v8; HISTORY v1 → v2 |
| E2 | PROMOTE v6 | PASS | `NOT PROMOTED v6: does not beat production v2: CI low 11% <= 38%` |
| E3 | FORCE v4 | PASS | `NOT PROMOTED v4: latency p95 7338 ms > budget 500 ms` |
| E4 | REPORT on v7 | PASS | REPORT tab, model `cathlab-pick@v7`, its 3 eval reports selected |
| E5 | REGISTER (form) | PASS | `REGISTERED cathlab-pick@v9` |
| E6 | ROLLBACK | PASS | `ROLLED BACK: production is v1`; HISTORY `ROLLBACK v1 rolled back from v2`; then `make demo-reset` → production v2 |
| E7 | REFRESH | PASS | table reloads |
| F1 | CROSS-PHYSICS | PASS | 8 rows × 3 engines, in-domain bold; `xvla_panda_pick_v2` (3 engines) 13 / 13 / 17 % all bold |
| F2 | Select `rep-cathlab-pick-v2` | PASS | 38 / 0 / 0 %; `2 train-split failing cells, 16 targeted episodes`; CREATE NEXT RUN enabled |
| F3 | CREATE NEXT RUN | PASS | `NEXT RUN … CREATED`; PIPELINE opens, Generate `split=cells:…/rep-cathlab-pick-v2/next_run.json` |
| F4 | BUILD REPORT for v7, k=4 | PASS | `REPORT rep-… READY`; listed (genesis 17 · maniskill 13 · robosuite 13 %, 1 cell) |
| G1 | Open ASSIST | PASS | `coronary_angiography`, 6 steps |
| G2 | Step access_prep | PASS | `Shall I hand you the vial?  (needed now)`, instruction `Hand me the vial`; YES/NO enabled |
| G3 | Step sheath_insertion | PASS | `Shall I hand you the bottle? (needed for the next step 'contrast_injection')` |
| G4 | YES with production model | PASS | `HANDOVER VIAL STARTED`; SmolVLA `success in 149 steps`; video shown |
| G5 | YES with scripted expert (step anaesthetic_injection → syringe) | PASS | expert `success in 154 steps`, GIF shown — the demo fallback |
| G6 | NO | PASS | proposal cleared, YES disabled |
| G7 | RECORDED ROLLOUTS → REFRESH | PASS | 16 cards: production videos from `configs/demo_videos.yaml` (e.g. `SmolVLA v2 hands over the vial (robosuite) ✓`) + local rollouts |
| H1 | AGENT: click a template | PASS | form with fields + `__extra` + agent select; Start agent present (not started) |
| H2 | AGENT: open a job | N/A | no agent jobs on this machine |
| H3 | AGENT BRIEF: "Add a MuJoCo engine" + engine gate | PASS | `# Agent brief: Add a MuJoCo engine`; COPY, DOWNLOAD .MD, RUN WITH AGENT enabled |
| H4 | COPY / DOWNLOAD .MD / CLOSE | PASS (CLOSE); manual check for DOWNLOAD | drawer closes; no download event captured by automation |
| H5 | LOG | PASS | 16 jobs with kind, name, state, time |
