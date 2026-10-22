# awareness

Front-end that turns what the clinician is doing into a natural-language `instruction: str` for any VLA served by the pipeline. The robot side is only that string, so every policy gains the feature without retraining.

| Module | Role |
|---|---|
| `procedure.py` | `ProcedureSpec`: ordered cathlab steps and the tools each needs (`configs/procedures/coronary_angiography.yaml`). `next_tool(step)` is the next tool a later step needs |
| `vlm.py` | `VLMClient` interface; `ScriptedVLM` (tests/offline), `GeminiVLM` (lazy; needs `google-genai`, `pillow`, `GOOGLE_API_KEY`) |
| `engine.py` | `AwarenessEngine.observe(frame, t) -> AwarenessState(step, confidence, next_tool, rationale, t)`. The VLM only names the current step; the step may stay or advance by one, and low-confidence readings are ignored |
| `assistant.py` | `Assistant.run(stream) -> Iterator[str]`: propose the next tool, ask for confirmation, emit the instruction. Cooldown, no re-offer of an accepted tool, no re-offer of a declined tool within the same step |

```python
procedure = ProcedureSpec.from_yaml("configs/procedures/coronary_angiography.yaml")
engine = AwarenessEngine(procedure, GeminiVLM())
assistant = Assistant(engine, task, confirm=ask_yes_no, speak=say)
for instruction in assistant.run(frames):  # frames yield (t, HxWx3 uint8)
    policy_client.send(instruction)
```

`task` is anything with `instructions(obj) -> list[str]` (a `TaskSpec`). `confirm` and `speak` are plain callables, so voice (ElevenLabs) plugs in without changes here.

Tests: `uv run pytest tests/test_awareness_*.py` (no network, GPU or Isaac).
