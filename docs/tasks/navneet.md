# Navneet — Awareness (Proactive Assist) and Voice

> **Update (2026-10-23 H24): Isaac Sim is dropped** (no working run after a day). Engines now: robosuite and ManiSkill3 (Panda), Genesis planned, FR3 Duo via Aravind's MuJoCo/Newton track. Isaac items below are historical.

> **Update (2026-10-23): Navneet owns the Isaac Sim engine**, built on the RTX 4090 PC: follow [`docs/handoffs/2026-10-23-isaac-engine-session.md`](../handoffs/2026-10-23-isaac-engine-session.md) (Isaac as an engine plugin: preview → check-expert → generate → convert → eval → rollout, FR3 Duo right arm, Siemens 16-D schema). It runs in parallel with the awareness work below. In Part B, awareness feeds the Control Station's ASSIST panel ([spec](../specs/2026-10-23-multi-engine-platform-design.md) P11).

> **Update (H6.5):** the task objects are now **bottle, vial, syringe** (`configs/tasks/cathlab_pick.yaml`).
> - Map procedure steps to these, e.g. access prep → vial (local anaesthetic), contrast → bottle, flush/injection → syringe.
> - `validate_against(task)` should use `cathlab_pick.yaml`.

## Agent brief (paste to your agent)

> You are helping build the CathLab Companion learning loop (spec: `docs/specs/2026-10-22-cathlab-learning-loop-design.md`, §5.1).
>
> My scope is `awareness/`: a front-end that produces natural-language instructions for any VLA served by the pipeline:
> - (1) **voice**: ElevenLabs speech-to-text (Scribe) turns a microphone/audio clip into an instruction, and ElevenLabs TTS speaks confirmations;
> - (2) **proactive assist**: a VLM watches the scene camera (co-human hands/pose, objects in use) and a procedure-step model (cathlab step state machine + VLM step recognition) predicts the next needed tool and proposes it ("Shall I hand you the sheath?"); on confirmation it emits `"Hand me the sheath"` to the policy server.
>
> The interface to robots is only `instruction: str` sent via `serve.client.PolicyClient`, so every VLA gains the feature without retraining. API keys come from `.env` and are never committed. Write tests with recorded audio/images and mocked API calls.

## Deliverables

| # | Deliverable | Path | Due |
|---|---|---|---|
| N1 | Cathlab procedure model: ordered steps (e.g. access → sheath insertion → guidewire → catheter → contrast/syringe → closure) with the tools each step needs, as YAML | `configs/procedures/coronary_angiography.yaml` | H6 |
| N2 | `voice.transcribe(audio) -> str`, `voice.to_instruction(text, task: TaskSpec) -> str \| None` (maps free speech to a valid object), `voice.speak(text)` | `awareness/voice.py` | H10 |
| N3 | `AwarenessEngine.observe(frame, t) -> AwarenessState(step, confidence, next_tool, rationale)` using a VLM (Gemini / Qwen-VL / GPT-4o class; pick by latency) | `awareness/engine.py` | H18 |
| N4 | Proactive loop: propose → confirm (voice yes/no) → emit instruction; with a cooldown and confidence threshold to avoid nagging | `awareness/assistant.py` | H22 |
| N5 | Demo scene: a co-human proxy in sim (animated hand/marker or pre-recorded clip of a clinician at a step) + live voice demo | `awareness/demo.py` | H26 |
| N6 | Stretch: an "anticipation dataset" recipe that pairs human-context frames with tool choice, so a VLA could learn proactivity directly (concept slide) | `docs/research/anticipation.md` | H30 |

## Interfaces

- **Consumes:** `TaskSpec` (object vocabulary), `serve.client.PolicyClient` (Sharan) or a direct `PolicyAdapter`; scene frames from a `SimBackend.render()`.
- **Produces:** `AwarenessState`, `Assistant.run(stream) -> Iterator[str]` (instructions).

## Acceptance

- Saying "Can you pass me the guidewire?" makes the sim robot execute the guidewire handover.
- From a frame or clip showing the "sheath insertion" step, the system proposes the guidewire before being asked, and logs its rationale.
- Latency (speech end → instruction) is measured and reported.
