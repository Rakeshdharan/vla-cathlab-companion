# Handover: Navneet's GPU machine (Isaac Sim) session

*Written 2026-10-22 on Navneet's Mac, for the Claude Code session that will run on the GPU PC. Say "follow `docs/handoffs/2026-10-22-navneet-gpu-machine.md`" as the first message.*

## Context

Parent handoff: [`2026-10-22-isaac-fr3-session.md`](2026-10-22-isaac-fr3-session.md). Read it first: it holds the decisions, the verified USD facts, and tasks I0 to I6.

Navneet's own workstream (awareness) is done through N4 and merged. This session is the shared Isaac push, on a machine that can actually run Isaac Sim.

## Machine setup (do once)

```bash
git clone git@github.com:Rakeshdharan/vla-cathlab-companion.git && cd vla-cathlab-companion
git config core.hooksPath .githooks            # strips AI co-author trailers
git config user.name "Navneet Kishan Srinivasan" && git config user.email "navneetkishan54@gmail.com"
uv sync
```

Then:
1. Put the provided `Hackathon_FR3Duo_flattened.usd` somewhere outside git (the team file share, or Rakesh's HF repos; see `docs/data.md`). Note its absolute path.
2. Record the Isaac Sim version (4.5 uses `omni.isaac.*`, 5.x uses `isaacsim.*`) and which Python runs it. This is task I0's first bullet and nobody has logged it in `docs/decisions.md` yet.
3. Confirm headless works: `SimulationApp({"headless": True})`, load the USD, step 100 frames, save one image per camera.
4. Run real-sim tests only under `@pytest.mark.gpu`. The default `uv run pytest` must stay GPU-free.

Commits carry Navneet's name only. Never commit `.env`, USD, datasets or checkpoints.

## What is already merged on `main`

| Item | Where |
|---|---|
| Awareness N1, N3, N4 (procedure model, VLM engine, propose-confirm-emit assistant) | `awareness/`, `configs/procedures/` |
| **I1 objects**: `setup_objects(stage)` for bottle (0.30 kg), vial (0.02 kg), syringe composite (0.015 kg), box parked | `sim/isaac/objects.py`, `tests/test_isaac_objects.py` |

I1 was written without Isaac or the real USD, so it rests on prim paths and sizes copied from the handoff. Its 11 tests run on an in-memory stage with `usd-core`.

## Order of work on the GPU machine

1. **Verify I1 against the real USD** (30 min). This is the first thing to do.
   - Load the USD, call `setup_objects(stage)`, and fix any missing prim path or wrong size.
   - Step 300 frames. Pass = all three objects rest on cart `SM_SupplyCart_01e` with drift ≤ 2 mm and no jitter.
   - If the syringe tunnels or jitters, enable CCD and consider a thicker collider.
   - Save before/after images, and record any deviation in the PR description.
2. **I0 reach probe** (Mitesh's task, but it needs this machine; check with Mitesh first so it isn't done twice).
   - With differential IK, move `right_tcp` to the 4 corners of a candidate pick zone on cart 01e and to the tray centre, 5 cm above the cart top, gripper pointing down.
   - Pass = images saved, the zone fixed with all corners reachable within 1 cm, and the cart-top height measured (expected y ≈ 0.966).
   - Log the pick-zone corners in world coordinates in `docs/decisions.md`.
3. **I5 variation**: `sim/isaac/randomize.py` with `apply_episode_config(stage, spec, cfg)` (object poses, colours, cart colour, mass, friction, lights). It needs the pick zone from step 2 and the shared `dataengine/variation.py` (see blockers).
   - Pass = a PNG grid of 8 visibly different configs.
4. **I3 expert and I4 recorder** are Mitesh's and Sharan's. Don't start them without coordinating.

## Blockers and things to check first

- **Shared dataengine code is not on `main` yet.** `dataengine/variation.py`, `expert.py`, `geometry.py` and `configs/variation/cathlab_v1.yaml` exist only on `origin/feat/rakesh-robosuite-base` (23d6181). Run `git fetch && git log origin/main -- dataengine/variation.py` to see if it has landed. If not, ask Rakesh, or branch from that branch knowing it carries Rakesh's commits (a PR would then include them, so wait for the merge where possible).
- **`contracts/__init__.py` on `main` doesn't export `TaskSpec`**, and `uv run pytest` fails at collection on 4 files. It's Mitesh's data-engine stub overwriting the real exports. Raise it with Rakesh. Don't fix it in an unrelated PR.
- **`configs/tasks/cathlab_pick.yaml` is missing from `main`** (it's on Rakesh's branch). Awareness tests use a stand-in object list until it lands.
- **`usd-core` is not a declared dependency.** I1 tests skip without it. Install with `uv run --with usd-core pytest tests/test_isaac_objects.py`, or ask Rakesh to add an `isaac` extra.
- **`pxr` and Isaac's bundled Python:** inside Isaac's own environment `pxr` is already present, so the I1 tests need no extra install there.

## Rules for this session

- Edit only files in the area you're assigned (`sim/isaac/` is shared here by the handoff). No changes to `contracts/` except through a separate PR that tags Rakesh.
- Small PRs on `feat/navneet-isaac-<topic>`, one task each. Fill in the PR template.
- Tests that need Isaac get `@pytest.mark.gpu` so CI stays green.
- Log non-obvious decisions in `docs/decisions.md` (one row each; include the real hour in the Time column).
- Report results (per-object stability, pick-zone corners, Isaac version, schema deviations) in the PR description, so the master session can read them.

## Open items on the awareness side (not for this session)

- N2 voice (ElevenLabs STT/TTS, `voice.py`) is due H10 and has no plan yet.
- `GeminiVLM` has never been run against the real API; latency isn't measured.
- N5 demo needs frames from a working sim plus `serve.client.PolicyClient`.
