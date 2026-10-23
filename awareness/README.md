# awareness

Front-end that turns what the clinician is doing into a natural-language `instruction: str` for any VLA served by the pipeline. The robot side is only that string, so every policy gains the feature without retraining.

| Module | Role |
|---|---|
| `procedure.py` | `ProcedureSpec`: ordered cathlab steps and the tools each needs (`configs/procedures/coronary_angiography.yaml`). `next_tool(step)` is the next tool a later step needs |
| `vlm.py` | `VLMClient.classify(frames, procedure)` takes the last few frames. `ScriptedVLM` (tests/offline), `AnthropicVLM` (lazy; `anthropic`, `pillow`, `ANTHROPIC_API_KEY`) and `GeminiVLM`. `AnthropicVLM` has cost guardrails: `VLMBudget` (default 60 calls / 150k tokens, raises `BudgetExceeded` before the API call), at most 4 frames per call, 224 px JPEGs, 200 output tokens, and an in-memory cache so identical frames cost nothing |
| `engine.py` | `AwarenessEngine.observe(frame, t) -> AwarenessState(step, confidence, next_tool, rationale, t, observed_tool, corroborated)`. Keeps a sliding window of frames; `min_interval_s` limits how often the VLM is queried. The VLM names the current step (stay or advance by one, low confidence and unknown steps ignored); `next_tool` always comes from the state machine. A tool the VLM sees that belongs to the current step only raises confidence |
| `clips.py` | `load_clip` / `load_clips`: labelled frame folders (`labels.yaml` + `NN.png`) → `LabeledClip.stream(fps)` |
| `clipgen.py` | Synthetic demo clips: chained OpenAI image edits of one reference frame, labelled by construction (`generate_clip`). Output goes under git-ignored `var/awareness/clips/` |
| `voice.py` | ElevenLabs: `Speaker.speak(text) -> mp3 path` (flash model, 22 kHz mp3, ≤120 chars/phrase, disk cache in `var/awareness/tts`, `enabled=False` switch) and `Listener.transcribe(audio, seconds)` (Scribe). `VoiceBudget` (default 600 chars, 60 s) raises `BudgetExceeded` before the API call. `parse_yes_no(text)` → True/False/None. Needs `elevenlabs` (+ `python-dotenv`); key from `.env` |
| `monitor.py` | Local test page: `uv run --with pillow --with anthropic --with elevenlabs --with python-dotenv python -m awareness.monitor` → http://127.0.0.1:8765. Shows frames, each VLM reading with its rationale, engine state, the predicted object, the question with an audio player and the rollout params. `demo` mode is canned and free; `live` and `speak` are budgeted and only run when ticked |
| `assistant.py` | `Assistant.run(stream) -> Iterator[str]`: ask "Doctor, do you need the {tool}?" (`phrase`), wait for confirmation, emit a `Handover`: the instruction string ("Hand me the syringe", a trained template) that also carries `.tool`. `handover.rollout_params()` gives `{"object", "instruction"}`, the pair the rollout stage needs (`--object` sets the scene target and success check, `--instruction` is only the text the policy hears). Cooldown, no re-offer of an accepted tool, no re-offer of a declined tool within the same step |

```python
procedure = ProcedureSpec.from_yaml("configs/procedures/coronary_angiography.yaml")
engine = AwarenessEngine(procedure, GeminiVLM())
assistant = Assistant(engine, task, confirm=ask_yes_no, speak=say)
for instruction in assistant.run(frames):  # frames yield (t, HxWx3 uint8)
    policy_client.send(instruction)
```

`task` is anything with `instructions(obj) -> list[str]` (a `TaskSpec`). `confirm` and `speak` are plain callables, so voice (ElevenLabs) plugs in without changes here.

Tests: `uv run pytest tests/test_awareness_*.py` (no network, GPU or simulator).
