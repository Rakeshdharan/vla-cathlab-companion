# Session handover: finish the demo on the RTX 2000 Ada (frontend + review); RTX 6000 Ada keeps the GPU jobs

*2026-10-24 ~04:00 (≈ H37.5). From the RTX 6000 Ada session (Navneet). Paste into the agent on the RTX 2000 Ada:
"follow `docs/handoffs/2026-10-24-rtx2000-demo-finish-session.md`". Everything below is on `main`; nothing in `var/`
is needed from the RTX 6000 Ada, because `make demo-bootstrap` rebuilds it from git.*

## 1. Where things stand

| Area | State |
|---|---|
| Datasets | `pick_robosuite@v1` (384), `pick_maniskill@v1` (369), `pick_genesis@v1` (383, + camera/light axes), FR3 `cathlab_fr3_newton_v1@v1.0.1`; `pick_panda_v2` = local merge of the three Panda sets (1,136 eps, on the RTX 6000 Ada only) |
| Checkpoints | 6 Panda (ACT, SmolVLA ×2, X-VLA ×3) + 2 FR3 on HF; **`xvla_panda_pick_v2`** (cross-sim co-training) finishing on the RTX 6000 Ada ~04:20 |
| Cross-physics table | 21 cells, [`reports/robosuite/cross_physics_v1.md`](../../reports/robosuite/cross_physics_v1.md). Off-engine 0–2 %, except **X-VLA-Genesis → robosuite 18 %** |
| Registry | `make demo-bootstrap` → `cathlab-pick@v1..v6`, **production v2 = `smolvla_panda_pick_v1`** (38 % robosuite), failure report `rep-cathlab-pick-v2` + `next_run.json` |
| UI | PROJECT, PIPELINE (9 stages incl. train/register/promote), DATA, MODELS, REPORT (+ **cross-physics table**), ASSIST, AGENT, LOG, AGENT BRIEF |
| Tests / CI | `uv run pytest -q` 313 passed; GitHub Actions on 3.10/3.12; `docker/control.Dockerfile` |

## 2. What the RTX 6000 Ada still does (results land on `main`, no action needed here)

| When | Job | Output |
|---|---|---|
| ~04:20 | `xvla_panda_pick_v2` training ends, pushed to HF | `Rakeshdharan56/xvla_panda_pick_v2` |
| ~04:20–04:50 | Production rollout videos (robosuite, ManiSkill, Genesis × bottle/vial/syringe) + targeted generate of `next_run.json` cells in the 3 engines | `var/loop_v2/` there; videos uploaded to HF for the demo (link in STATUS) |
| ~04:50–05:10 | `xvla_panda_pick_v2` eval in robosuite + ManiSkill + Genesis | eval reports + regenerated cross-physics table committed to `main` |

After that commit: `git pull && make demo-bootstrap` here registers `xvla_panda_pick_v2` as v7 and tries to promote it by
the rule (it becomes production only if its robosuite CI low beats 38 %).

## 3. Setup on the RTX 2000 Ada (≈ 10 min)

```bash
cd ~/vla-cathlab-companion && git pull --rebase && git config core.hooksPath .githooks
uv sync && NO_PROXY=127.0.0.1,localhost uv run pytest -q          # 313 passed
make demo-bootstrap                                                 # registry + failure report into var/
make dev                                                            # UI on http://localhost:8700
# robosuite worker for PIPELINE rollouts and ASSIST (existing env from the cross-eval handoff):
source ~/envs/vla-eval/bin/activate && pip install fastapi uvicorn httpx
MUJOCO_GL=egl python -m orchestrator.worker --name robosuite --capabilities scene,generate,eval --port 8711
```

ManiSkill can't run here (WSL2 has no Vulkan); Genesis can, on CPU (`sim/genesis/README.md`, `CATHLAB_GENESIS_BACKEND=cpu`), but it is not needed for the demo.

## 4. Review (in this order, ~30 min)

| # | What | How | Look for |
|---|---|---|---|
| R1 | **REPORT → CROSS-PHYSICS** | open the tab | 7 rows × 3 engines, in-domain cells bold, numbers = `cross_physics_v1.md` |
| R2 | **MODELS** | after `make demo-bootstrap` | production v2; history v1 → v2; PROMOTE v6 is refused with "does not beat production v2: CI low 11 % ≤ 38 %" |
| R3 | **REPORT → `rep-cathlab-pick-v2`** | select it | robosuite 38 % / ManiSkill 0 % / Genesis 0 %, failing cells, 2 next-run cells; CREATE NEXT RUN prefills PIPELINE generate |
| R4 | **DATA** | open the tab | 5 published/local datasets with HF links; local generate outputs table (empty until a generate runs) |
| R5 | **PIPELINE** | new run (robosuite, all objects) | stages train / register / promote listed; register blocked "needs 'eval' to finish first"; train blocked unless a worker advertises `train` |
| R6 | **ASSIST** | robosuite worker running | proposes the object, YES → rollout with `registry:cathlab-pick@production` (SmolVLA) and a video |
| R7 | Code review of the H37 commits | `git log --oneline 51ddf44..origin/main` | esp. `dataengine/` (P6 rules: no-op drop, success hold, shard checks), Rakesh-owned |

Anything wrong in R1–R6: fix in `frontend/` / `orchestrator/` (tests in `tests/orchestrator/`), small commits to `main`.

## 5. Remaining tasks for the demo (pick in order)

1. **Demo script** (`docs/DEMO.md`): the 5-minute walk-through above (R1 → R2 → R3 → CREATE NEXT RUN → R6), the
   headline sentence ("single-engine policies don't transfer; data with visual variation is the first lever: 0 → 18 %"),
   and the fallback (pre-recorded rollout videos from §2) if a live rollout is slow.
2. **Rollout videos in the UI without a GPU worker:** a small section in REPORT or ASSIST that lists the uploaded
   production videos (HF URLs from STATUS) so the demo works offline from the GPU.
3. **After the v2 evals land:** `git pull && make demo-bootstrap`, check R1/R2 again (v7 row, maybe a new production).
4. **Optional:** DATA tab "load stats" button (episodes per object from `meta/episodes_extra.jsonl` of local outputs);
   `dataset_intel/` readiness score; ASSIST VLM step tracking (`awareness/vlm.py`) and voice.

## 6. Gotchas

- Proxy machines: `NO_PROXY=127.0.0.1,localhost` or localhost calls get 407 (4 core tests and the UI's worker lamps).
- `var/` is git-ignored: registry, reports, runs are per machine. `make demo-bootstrap` is idempotent.
- `configs/variation/cathlab_v1.yaml` is frozen (the harness guard rejects jobs that touch it).
- Commits as yourself, no AI `Co-authored-by` trailers (the hook strips them); rebase on `main` before pushing.
