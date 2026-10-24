# Session Handover: RTX 2000 Ada: Control Station user guide (wiki) + end-to-end click test

*From the RTX 6000 Ada session, 2026-10-24 ~07:30 (≈ H40). Owner: Rakesh. Paste into the agent on the RTX 2000 Ada, or say
"follow `docs/handoffs/2026-10-24-rtx2000-ui-guide-session.md`". Pitches today: 09:00 tech pitch (10 min, at our table),
12:00 jury pitch (4 min) — see [`../PITCH.md`](../PITCH.md).*

## 1. Goal

1. **Click through the whole Control Station** with the script in §4. Every step has the button to press and what must happen.
   Record PASS / FAIL + a one-line note per step.
2. **Write `docs/GUIDE.md`**, a wiki-style user guide (GitHub-wiki compatible: one H1, H2 per tab, tables, relative links):
   - one section per tab (PROJECT, PIPELINE, DATA, MODELS, REPORT, ASSIST, AGENT, LOG) and the AGENT BRIEF drawer + footer;
   - for **every button**: what it does, the API it calls, what you see after pressing it, and the common error messages;
   - one subsection per **PIPELINE stage** (§3): what it runs, where (worker), its parameters, what blocks it, its outputs;
   - a "first 15 minutes" quick start and a troubleshooting table (§5);
   - screenshots (optional, `docs/img/guide/*.png`, < 300 kB each).
3. **Fix only trivial UI bugs** found by the click test (labels, obvious JS errors), each with a test where the code has
   tests. Anything bigger: list it under "Known issues" in GUIDE.md and in `docs/STATUS.md`.
4. Link GUIDE.md from `README.md` (Doc table) and `docs/DEMO.md` (top), commit, push.

**Time box:** GUIDE.md first draft on `main` by **08:30** (before the 08:35 freeze, docs only), or right after the 09:00
pitch. The click test itself doubles as the 09:00 demo rehearsal.

## 2. Setup on the RTX 2000 Ada (WSL2, 8 GB; ≈ 10 min)

```bash
cd vla-cathlab-companion && git pull --rebase && uv sync
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost     # proxy machines: else 407 on every localhost call
make demo-bootstrap                 # var/registry + production failure report from the committed eval reports
make dev                            # terminal 1: control plane + UI on http://localhost:8700
```

Workers (one terminal each; env = the eval env with torch + lerobot + robosuite, e.g. `~/envs/vla-eval`):

```bash
# terminal 2: robosuite (scene, generate, eval, rollout)
MUJOCO_GL=egl ~/envs/vla-eval/bin/python -m orchestrator.worker --name robosuite --capabilities scene,generate,eval --port 8711
# terminal 3: lerobot (convert, intel, train). Do NOT run a full training on this laptop.
~/envs/vla-eval/bin/python -m orchestrator.worker --name lerobot --capabilities convert,intel,train,policy_server --port 8712
# optional terminal 4: Genesis (only if ~/venvs/genesis exists here)
CATHLAB_GENESIS_BACKEND=cpu PYTHONPATH=. ~/venvs/genesis/bin/python -m orchestrator.worker --name genesis --capabilities scene,generate,eval --port 8715
```

- **ManiSkill can't run here** (WSL2 has no Vulkan): its lamp stays red and its stages say `worker 'maniskill' is offline`.
  That is an expected result to document, not a bug.
- `pip install fastapi uvicorn httpx` in the worker env once, if `orchestrator.worker` fails to import.

## 3. Pipeline stages (from `orchestrator/stages.py`)

Each run (PROJECT → CREATE RUN) has 9 stages. A stage lamp is **ready** (green), **blocked** (grey, hover = reason),
**running**, **done** or **failed**. Blocking reasons come from `orchestrator/pipeline.py`:
`no engine plugin '<id>'`, `<engine>: '<capability>' is planned|in progress`, `worker '<name>' is offline`,
`no online worker with capability 'convert'|'train'`, `needs '<stage>' to finish first`, `next run file missing: …`.

| Stage | Runs | Where | Params (default) | Needs | Outputs (OUTPUTS list under the log) |
|---|---|---|---|---|---|
| **Scene preview** | `engines.<sim>.cli preview` | the run's engine worker | n (8) | — | `grid.png`: 7 train configs + 1 all-held-out |
| **Check expert** | `check-expert` | engine worker | n (5) episodes/object, split (train) | — | `expert_check.json`, per-object success in the log |
| **Generate data** | `generate` | engine worker | n (30), start (0), split (train or `cells:<next_run.json>`) | — | `raw/shard_*/episodes/*.npz`, `manifest.yaml`, `episodes_extra.jsonl` |
| **Convert to LeRobot** | `convert` | any worker with `convert` (lerobot) | — | generate | `lerobot/` LeRobot v3 dataset |
| **Evaluate** | `eval` | engine worker | policy (expert), model-version, tiers (in_distribution), max-seeds (1) | — | `eval_report.json`, failure videos |
| **Rollout** | `rollout` | engine worker | policy (expert), object, seed (5000), instruction | — | `rollout.mp4`/`.gif`, `rollout.json` |
| **Train policy** | `python -m train.recipe --run` | any worker with `train` | model (smolvla), dataset (required), repo-id (required), steps, io, dataset-root | — | checkpoint pushed to `repo-id` |
| **Register version** | control plane (registry) | local | ckpt, datasets | eval | new `cathlab-pick@vN` in MODELS |
| **Promote to production** | control plane (rule) | local | version (default: the one this run registered) | register | PROMOTED / NOT PROMOTED + reason |

Policy strings: `expert`, `random`, `lerobot:<hf-repo>`, `url:<policy-server>`, `registry:cathlab-pick@<vN|production>`.

## 4. End-to-end click test (the walkthrough; ≈ 45 min with live rollouts)

Status messages appear in the footer (bottom bar). Worker lamps are top right (green = healthy).

### A. Shell and header
| # | Do | Expect |
|---|---|---|
| A1 | Open http://localhost:8700 | Header `CATHLAB CS`, 8 tabs, AGENT BRIEF button, worker lamps; footer `READY`, ▶ RUN STAGE and ■ STOP disabled |
| A2 | Hover each worker lamp | robosuite + lerobot green (genesis green if started), maniskill red/offline |

### B. PROJECT (build a run)
| # | Do | Expect |
|---|---|---|
| B1 | Click each node in PROJECT TREE (simulator, embodiment, objects, expert, variation, model) | Items of that section with status badges (ready / in progress / planned) |
| B2 | simulator → `robosuite`; embodiment → `panda`; objects → bottle, vial, syringe; expert → `analytic`; model → `smolvla` | SELECTION CHECK shows no problems |
| B3 | Select a **planned** item (e.g. simulator `newton`) | SELECTION CHECK lists a problem; the item offers **Build with agent** |
| B4 | Click **Build with agent** on a planned item | Switches to AGENT with the matching template prefilled (don't start it) |
| B5 | Re-select `robosuite`, type a note, click **CREATE RUN** | Footer `RUN <id> CREATED`; CURRENT RUN shows the id + selection |

### C. PIPELINE (the 9 stages)
| # | Do | Expect |
|---|---|---|
| C1 | Open PIPELINE, pick the new run in RUN | Strip of 9 stages; preview/check-expert/generate/eval/rollout ready; convert blocked `needs 'generate'`; register blocked `needs 'eval'`; promote blocked `needs 'register'`; train ready only if a worker has `train` |
| C2 | Click a blocked stage | Detail: `BLOCKED: <reason>`; ▶ RUN STAGE disabled |
| C3 | **Scene preview** → ▶ RUN STAGE (n 4) | Footer `STARTED SCENE PREVIEW`; live log; lamp running → done; OUTPUTS `grid.png` thumbnail |
| C4 | **Check expert** (n 2) → RUN | `expert_check.json`; log shows per-object success (expect 100 %) |
| C5 | **Generate data** (n 3) → RUN | 3 episodes kept; outputs `raw/…npz`, `manifest.yaml`; then **Convert** becomes ready |
| C6 | **Convert to LeRobot** → RUN | `lerobot/` folder in OUTPUTS; DATA → LOCAL GENERATE OUTPUTS lists the run after REFRESH |
| C7 | **Evaluate**: policy `expert`, tiers `in_distribution`, max-seeds 1 → RUN | 3 episodes, success 100 %; `eval_report.json`; **Register** becomes ready |
| C8 | **Rollout**: policy `registry:cathlab-pick@production`, object `vial` → RUN | First use downloads the SmolVLA checkpoint; `rollout.mp4` plays in OUTPUTS (success expected on vial seed 5000/5001) |
| C9 | During any running stage press ■ STOP | Footer `STOP SENT`; lamp → failed/cancelled |
| C10 | **Register version**: ckpt `Rakeshdharan56/smolvla_panda_pick_v1`, datasets `Rakeshdharan56/pick_robosuite@v1` → RUN | Finishes at once; MODELS shows a new version |
| C11 | **Promote to production** → RUN | NOT PROMOTED (a 3-episode eval is < 30 episodes) with the reason in the log — expected |
| C12 | **AGENT BRIEF FOR THIS STAGE** (any stage detail) | Drawer opens with that stage + engine ticked |
| C13 | **Train policy** (only check, don't run on 8 GB) | Params form shows model/dataset/repo-id/steps/io/dataset-root; blocked if no `train` worker |

### D. DATA
| # | Do | Expect |
|---|---|---|
| D1 | Open DATA | PUBLISHED DATASETS from `configs/datasets.yaml` with HF links (pick_robosuite, pick_maniskill, pick_genesis, pick_panda_v2, FR3 Newton) |
| D2 | **REFRESH** | LOCAL GENERATE OUTPUTS lists C5's output (episodes, objects) |

### E. MODELS (registry)
| # | Do | Expect |
|---|---|---|
| E1 | Open MODELS | PRODUCTION `cathlab-pick@v2`; promotion-rule sentence; VERSIONS table v1…v7 (+ your C10 version) with success per engine; HISTORY |
| E2 | **PROMOTE** on v6 (X-VLA Genesis) | Footer `NOT PROMOTED v6: does not beat production v2: CI low 11 % ≤ 38 %` |
| E3 | **FORCE** on v4 | Refused on latency (`p95 … ms > 500`): FORCE never skips latency |
| E4 | **REPORT** on a row | Jumps to REPORT with model label + that version's eval reports prefilled |
| E5 | **REGISTER** (left form): adapter `lerobot`, checkpoint + datasets, tick eval reports | Footer `REGISTERED cathlab-pick@vN`; new row |
| E6 | **ROLLBACK** → confirm (only if you promoted something) | Footer `ROLLED BACK: production is …`; HISTORY shows ROLLBACK. **Run `make demo-bootstrap` (or `make demo-reset`) afterwards so production is v2 again for the pitch** |
| E7 | **REFRESH** | Table reloads |

### F. REPORT (cross-physics + failure reports + next run)
| # | Do | Expect |
|---|---|---|
| F1 | Open REPORT, **REFRESH** (CROSS-PHYSICS) | 8 rows × 3 engines (robosuite, ManiSkill, Genesis) with 95 % CIs, in-domain bold; numbers = `reports/robosuite/cross_physics_v1.md` (+ v3 row once committed) |
| F2 | Click `rep-cathlab-pick-v2` in REPORTS | REPORT pane shows markdown: 38 / 0 / 0 %, failing cells, sim2sim gap, 2 next-run cells; **CREATE NEXT RUN** enabled |
| F3 | **CREATE NEXT RUN** | Footer `NEXT RUN <id> CREATED`; PIPELINE opens on the new run with Generate's split = `cells:…next_run.json` |
| F4 | New report: model label `cathlab-pick@v7`, tick its eval reports, k 4, **BUILD REPORT** | Footer `REPORT <id> READY`; appears in REPORTS |

### G. ASSIST (awareness → handover)
| # | Do | Expect |
|---|---|---|
| G1 | Open ASSIST, procedure `coronary_angiography` | 6 steps listed |
| G2 | Click step **access_prep** | PROPOSAL `Shall I hand you the vial? (needed now)` + instruction; YES/NO enabled |
| G3 | Click a step with no tool (e.g. sheath_insertion) | Proposal names the next needed tool or "nothing to hand over" |
| G4 | run = your robosuite run, policy = **production model (registry)**, **YES, HAND IT OVER** | Footer `HANDOVER VIAL STARTED`; log streams; video appears when done (few minutes on 8 GB) |
| G5 | Same with policy **scripted expert** | Always succeeds — the demo fallback |
| G6 | **NO** | Proposal cleared, nothing started |
| G7 | RECORDED ROLLOUTS → **REFRESH** | the production rollout videos (3 engines × 3 objects, from `configs/demo_videos.yaml`) + local rollouts; they play offline |

### H. AGENT, AGENT BRIEF, LOG
| # | Do | Expect |
|---|---|---|
| H1 | AGENT: click a template | Form with its fields; **Start agent** (don't start unless Copilot CLI is logged in; `make demo-ui` uses a fake agent) |
| H2 | AGENT: click an existing job (if any) | PROMPT / DIFF / PUSH BRANCH / DISCARD; console shows the job |
| H3 | **AGENT BRIEF** (header) → type "Add a MuJoCo engine", tick stage `engine` → **BUILD BRIEF** | Markdown brief in the drawer; COPY, DOWNLOAD .MD, RUN WITH AGENT enabled |
| H4 | COPY / DOWNLOAD .MD / CLOSE | Clipboard / `.md` file / drawer closes |
| H5 | LOG | JOBS table with every stage/handover/agent job, state and time |

## 5. Troubleshooting table for GUIDE.md (seed)

| Symptom | Cause | Fix |
|---|---|---|
| Every call fails, lamps red, footer 407 | HTTP proxy catches localhost | `export NO_PROXY=127.0.0.1,localhost` before `make dev` and the workers |
| Stage `worker '<x>' is offline` | worker not started / wrong port | start it (§2); ports 8711 robosuite, 8712 lerobot, 8714 maniskill, 8715 genesis |
| ManiSkill stages always offline here | WSL2 has no Vulkan | run ManiSkill on the RTX 6000 Ada; expected |
| Production is not v2 | leftovers from earlier tests | `make demo-reset` |
| First rollout slow | HF checkpoint download | run one rollout before the demo (cached afterwards) |
| Live rollout too slow for the demo | 8 GB laptop | ASSIST → RECORDED ROLLOUTS, or policy = scripted expert |

## 6. Rules

- Stay in `docs/` (GUIDE.md, images), `README.md`/`docs/DEMO.md` links, and trivial `frontend/` fixes with a note.
- Don't touch `var/registry` by hand; use the UI or `make demo-bootstrap`/`demo-reset`.
- Commit as yourself (no AI `Co-authored-by` trailers), `git pull --rebase` before pushing. Freeze 08:35–09:15.
