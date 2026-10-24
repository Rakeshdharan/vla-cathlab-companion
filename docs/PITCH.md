# Pitch plan: 09:00 tech pitch (10 min, at our table) · 12:00 jury pitch (4 min, slides)

Numbers are from `main` (cross-physics table [`reports/robosuite/cross_physics_v1.md`](../reports/robosuite/cross_physics_v1.md),
[`STATUS.md`](STATUS.md), [`decisions.md`](decisions.md)). v3 (loop retrain) is in: robosuite 32 %, ManiSkill 13 %, Genesis 22 % (v2: 13 / 13 / 17 %).

## Timeline (Saturday)

| Time | What | Who / where |
|---|---|---|
| ~08:00 | `xvla_panda_pick_v3` (loop retrain) finishes + pushes | RTX 6000 Ada, automatic |
| ~08:00–08:20 | v3 eval in robosuite + ManiSkill + Genesis (parallel), table rebuilt, registered `cathlab-pick@v8`, promote attempt | RTX 6000 Ada, automatic |
| 08:20–08:35 | Commit the v3 row; `git pull && make demo-bootstrap` on the demo laptop; one cached rollout (vial) | demo laptop |
| **08:35** | **Freeze**: no more pushes to `main` until after the 09:00 pitch | everyone |
| 08:35–08:55 | Rehearse the 10-min flow below twice, with the fallbacks open in tabs | presenters |
| **09:00** | **Tech pitch, 10 min, at our table** | |
| 09:15–11:30 | Build the 4-min deck (outline below), rehearse to 3:45 | |
| **12:00** | **Jury pitch, 4 min, slides** | |

Dropped for time: the Genesis no-axes ablation (data generated, 384/384, not trained), SmolVLA v2 co-training, FR3 eval
in this repo (Aravind's stack).

## 09:00 tech pitch: 10 minutes at the table

Show the Control Station (`make dev`, http://localhost:8700); every claim has a tab or a file behind it.

| Min | Topic | Show | Say (key numbers) |
|---|---|---|---|
| 0:00–1:00 | **Problem + task** | README diagram | Franka Panda hands cath-lab items (bottle, vial, syringe) on "Hand me the …". Sim-first, self-improving loop: ingest → dataset intel → fine-tune → eval → version & serve → failures become the next data. |
| 1:00–2:30 | **Platform** | PROJECT tab, `engines/` | One contract layer (`contracts/`), **3 physics engines as plugins** with the same 6 CLI stages: robosuite/MuJoCo, ManiSkill3/PhysX 5, Genesis. Shared variation grid (`cathlab_v1`: 384 train cells, 5 eval tiers) and shared scripted expert ⇒ **paired episodes across engines**. Coding-agent harness builds new engines/objects from a brief. |
| 2:30–3:30 | **Data** | DATA tab | `pick_robosuite` 384 eps / 69k frames, `pick_maniskill` 369 / 45k, `pick_genesis` 383 / 66k (+ camera/light jitter), merged `pick_panda_v2` 1,136 / 181k; FR3 Duo `cathlab_fr3_newton_v1` 384 / 160k. All public on HF. `ingest` contract check + `dataset_intel` readiness (robosuite 1.00, ManiSkill 0.97, provided FR3 84ep 0.98). |
| 3:30–5:30 | **Headline: cross-physics eval** | REPORT → CROSS-PHYSICS | Expert 100 % in all 3 engines. Single-engine policies: SmolVLA 38 % robosuite / 43 % ManiSkill **in-domain, 0 % off-engine**; failures at *reach*. Only exception: X-VLA trained on Genesis (visual variation) → 18 % in robosuite. **Co-training on all 3 engines (X-VLA v2): 13 / 13 / 17 % — the first single policy that works in every engine.** |
| 5:30–7:00 | **Version & serve** | MODELS tab | Registry `cathlab-pick@v1…v8`, promotion rule = 95 % CI low beats production, ≥ 30 episodes, p95 ≤ 500 ms. Production = v2 (SmolVLA, 38 %); v6/v7 refused by the rule. Policy server + `url:` client; ACT 20 ms, X-VLA 214 ms, SmolVLA 271 ms p95 per chunk. |
| 7:00–8:30 | **Close the loop** | REPORT → `rep-cathlab-pick-v2` → CREATE NEXT RUN | Failure report → `next_run.json` (failing **train-split** cells only) → targeted generate in all 3 engines (16/16 each) → retrain → **v3: robosuite 13 → 32 %, Genesis 17 → 22 %** from 48 targeted episodes → eval → registry (v8 refused: CI low 21 % ≤ 38 %, the rule stays strict). All automated, tested end to end on real checkpoints. |
| 8:30–9:30 | **Live: ASSIST** | ASSIST tab, step "access_prep", YES | Procedure model proposes the next tool ("Shall I hand you the vial?"), YES rolls out the production model live in robosuite. Fallback: RECORDED ROLLOUTS. |
| 9:30–10:00 | **Takeaway** | — | "Single-engine policies don't transfer; visual variation and cross-engine co-training are the levers, and the loop measures that automatically." |

Fallbacks: `DEMO.md` §3 (no worker → RECORDED ROLLOUTS; scripted expert always succeeds).

## 12:00 jury pitch: 4 minutes, 6 slides

| # | Slide | Content | Time |
|---|---|---|---|
| 1 | **The problem** | Cath-lab staff hand instruments under time pressure; a robot assistant needs policies that survive the move from simulation to reality. | 0:30 |
| 2 | **Our solution** | Simulation-first, self-improving loop (diagram): ingest → dataset intelligence → fine-tune VLAs → evaluate in 3 physics engines → version & serve → failures become the next data. | 0:45 |
| 3 | **What we built in 36 h** | 3 physics engines as plugins, 4 public datasets (~340k frames) + a merged co-training set, 10 trained checkpoints (ACT, SmolVLA, X-VLA; Panda + FR3 Duo), Control Station UI with registry, promotion rule, policy server, failure reports, coding-agent harness. | 0:45 |
| 4 | **Key result** | Cross-physics table as a heat map: in-domain only, 0 % off-engine; co-training on 3 engines → first policy working everywhere (13 / 13 / 17 %). | 0:45 |
| 5 | **The loop closes itself** | Failure report → 48 targeted episodes → retrain (v3): robosuite 13 → 32 %, Genesis 17 → 22 % → promotion rule decides. Screenshot of REPORT + MODELS. | 0:30 |
| 6 | **Next** | More visual variation + co-training, FR3 Duo bimanual handover, awareness (VLM step tracking + voice) proposing tools before they're asked for. | 0:30 |

Assets to grab before 11:00: the cross-physics table (REPORT tab screenshot), MODELS screenshot, one success + one failure
rollout video (`configs/demo_videos.yaml`), the architecture diagram from the README.
