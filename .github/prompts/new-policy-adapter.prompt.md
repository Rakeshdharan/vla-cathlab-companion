---
description: "Scaffold a new PolicyAdapter (VLA or baseline model) with a GPU-free test."
argument-hint: "model name, e.g. smolvla, pi05, groot, xvla, act"
agent: "agent"
---
Add a `PolicyAdapter` for the model `${input:model}`.

- File: `train/adapters/${input:model}.py`, a subclass of `contracts.PolicyAdapter`, implementing `load(ckpt)`, `predict(obs, instruction) -> np.ndarray (chunk_size, action_dim)` and `meta() -> PolicyMeta`.
- Import the model library lazily inside `load`. Map observation keys and action dims using `configs/robots/fr3_duo.yaml`.
- Register it in `train/adapters/__init__.py` under the key `"${input:model}"`.
- Test in `tests/train/test_${input:model}_adapter.py`: verify the registry entry and `meta()` without loading weights; mark a real-weights smoke test with `@pytest.mark.gpu`.
- Add a line to `docs/decisions.md` noting any action-space adaptation (e.g. bimanual masking).
