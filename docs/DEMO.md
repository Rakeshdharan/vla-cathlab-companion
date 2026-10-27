# Demo: 5-minute walk-through of the Control Station

> Full UI reference: [GUIDE.md](GUIDE.md) (every tab, button and stage; troubleshooting).

Audience: reviewers who have not seen the Control Station before. Covers REPORT → MODELS → REPORT
(failure report) → CREATE NEXT RUN → ASSIST. For the full UI and stage list see [README.md](../README.md);
for the pipeline internals see [docs/HARNESS.md](HARNESS.md) and [docs/EXTENDING.md](EXTENDING.md).

## 1. Before the demo (setup checklist, ~10 min)

```bash
git pull --rebase
uv sync
make demo-bootstrap     # rebuild var/registry + the production failure report from committed eval reports
make dev                 # UI on http://localhost:8700
```

For a live robosuite rollout in PIPELINE/ASSIST, start the worker (separate terminal, same machine):

```bash
source ~/envs/vla-eval/bin/activate
pip install fastapi uvicorn httpx   # once per env
MUJOCO_GL=egl python -m orchestrator.worker --name robosuite --capabilities scene,generate,eval --port 8711
```

That env already has torch, lerobot and robosuite.

| Check | Why |
|---|---|
| `var/registry` has no leftovers from earlier test runs | `make demo-bootstrap` keeps whatever is already in `var/registry`, so an older promotion can leave a different model as production. For a clean run: `rm -rf var/registry var/reports && make demo-bootstrap`. Production must come out as `cathlab-pick@v2`. |
| On a proxy machine, export `NO_PROXY=127.0.0.1,localhost` | otherwise localhost calls (UI ↔ control plane ↔ worker) get HTTP 407 |
| Run one rollout before the demo (ASSIST → YES, any object) | the policy checkpoint downloads from Hugging Face on first use; pre-caching it avoids a live download during the demo |

## 2. The 5-minute script

| Time | Tab | Click | What to say |
|---|---|---|---|
| 0:00–1:00 | **REPORT** → **CROSS-PHYSICS** | open the tab (loads on click) | "9 rows, 3 simulation engines (robosuite/MuJoCo, ManiSkill/PhysX 5, Genesis). Bold cells are in-domain: the expert hits 100% everywhere; every learned policy is strong only where it trained — ACT 7%, SmolVLA 38%, X-VLA 8% in robosuite — and drops to 0–2% off-engine. The one exception is X-VLA trained on Genesis: 18% back in robosuite, 27% in Genesis. Training on all three engines (v2) gives 13/13/17%, and one loop iteration on the failure report's cells (v3) lifts it to 32/13/22%." |
| 1:00–1:30 | same table | point at the X-VLA-Genesis row | Read the headline sentence: **"single-engine policies don't transfer; data with visual variation is the first lever: 0 → 18%."** Genesis renders with extra camera jitter and light variation; that's the difference. |
| 1:30–2:30 | **MODELS** | scroll the version table | "Production is `cathlab-pick@v2`, SmolVLA at 38% in robosuite — the best in-domain policy we have. v1 was the first production model (the scripted-free ACT baseline), replaced once SmolVLA's CI low (27%) beat it (7%). v6, X-VLA trained on Genesis, was refused: `does not beat production v2: CI low 11% <= 38%`." |
| 2:30–3:00 | same tab | point at the promotion rule line | "The rule: 95% CI low on robosuite has to beat production, with at least 30 episodes and p95 latency under 500 ms per predict call. FORCE can skip the success check but never the latency check — v4 was refused on latency alone, p95 7338 ms." |
| 3:00–4:00 | **REPORT** → select `rep-cathlab-pick-v2` | click the report in the list | "This is the failure report for production: 38% robosuite / 0% ManiSkill / 0% Genesis, with the failing cells and recommendations per cell." |
| 4:00–4:15 | same report | click **CREATE NEXT RUN** | "This switches to PIPELINE with GENERATE DATA prefilled: 2 held-out train-split cells, 16 targeted episodes. Held-out cells are never trained on — only train-split failures feed the next run." |
| 4:15–5:00 | **ASSIST** | pick procedure `coronary_angiography`, click step **"1. access_prep"**, policy **production model (registry)**, then **YES** | "`Shall I hand you the vial? (needed now)` — click YES and it rolls out `lerobot:Rakeshdharan56/smolvla_panda_pick_v1` live in robosuite, about 444 ms per predict on this GPU, a few minutes per episode, video appears in the tab." |

Use the **vial** for the live rollout: SmolVLA solved it in eval on all 4 in-distribution seeds (124–168 steps).
If time is short, switch the policy dropdown to **scripted expert** instead — it always succeeds (bottle in 198
steps) and is the safe fallback described below.

## 3. Fallbacks

| Situation | What to do |
|---|---|
| No GPU / robosuite worker not running | Stay on REPORT and MODELS (steps 1–4 of the script need no worker). For ASSIST, open **RECORDED ROLLOUTS** instead of clicking YES: it lists the production rollout videos uploaded from the RTX 6000 Ada session (`configs/demo_videos.yaml`) plus every finished rollout on this machine. |
| Live rollout is slow or stuck | Switch to **RECORDED ROLLOUTS** and play the matching video instead of waiting on the live job. |
| Network is down | Hugging Face checkpoint downloads need network the first time a policy is used. Run one rollout before the demo (see checklist) so the checkpoint is already cached locally; after that, live rollouts work offline. If nothing is cached, use **RECORDED ROLLOUTS**, which only reads local files. |
| ASSIST grasp fails live (e.g. the syringe, step 2) | Switch the policy dropdown to **scripted expert** and click YES again — it is a safe fallback with 100% success — or fall back to a recorded video. |

## 4. Questions we expect

**Why is cross-engine success 0% for most policies?**
Each policy learns the visual and physics quirks of the one simulator it trained on (shading, contact dynamics,
camera placement). Move it to a different engine and those cues no longer match what it saw in training, so it
fails at the same early stages (mostly at reach/grasp — see the per-stage table in
[reports/robosuite/cross_physics_v1.md](../reports/robosuite/cross_physics_v1.md)). X-VLA trained on Genesis is
the one case with real cross-engine transfer (18% in robosuite), because Genesis training data already varies
camera and light per episode.

**Why was v6 (X-VLA, Genesis) refused for production?**
The promotion rule requires the candidate's 95% CI low on robosuite to beat production's success rate. v6's CI
low was 11%, production (v2) is 38%: `does not beat production v2: CI low 11% <= 38%`. v4 was refused for a
different reason — its p95 latency (7338 ms) is far over the 500 ms budget, and FORCE cannot waive latency.

**What does the next run do?**
`CREATE NEXT RUN` targets only the train-split cells where production (v2) failed: 2 cells × 8 episodes = 16
episodes, generated and converted through PIPELINE, then retrained and re-evaluated. Held-out cells are excluded
by design, so the number reported for held-out tiers stays a fair test of generalization.

**How do I add a new engine or object?**
A new engine is a plugin under `engines/<name>/` implementing the same CLI contract (preview, check-expert,
generate, convert, eval, rollout) as `engines/robosuite`; a new object is added to the variation spec the engine
reads from. Anything the UI doesn't already expose — a new engine, a new object, a custom report — goes through
**Build with agent** (PROJECT tab) or the **AGENT BRIEF** drawer, which turns the request into a task for a
coding agent. See [docs/HARNESS.md](HARNESS.md) and [docs/EXTENDING.md](EXTENDING.md).
