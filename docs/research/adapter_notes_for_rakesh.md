# Adapter Notes for Rakesh — A1 Handoff

**From:** Akanksh · **Date:** 2026-10-22 H6  
**Purpose:** Concrete interface specs so Rakesh can implement `train/<model>/adapter.py`
without waiting on further VLA research decisions.

---

## What Rakesh Needs to Build

Three adapter files, one per locked-in model, each implementing the frozen `PolicyAdapter`
contract from `contracts/policy.py`:

```
train/
├── smolvla/
│   ├── adapter.py          ← SmolVLAAdapter (ready to copy from vla_selection.md)
│   └── configs/smolvla_v0.yaml
├── act/
│   ├── adapter.py          ← ACTAdapter (ready to copy from vla_selection.md)
│   └── configs/act_v0.yaml
└── diffusion/
    ├── adapter.py          ← DiffusionPolicyAdapter (optional, ready to copy)
    └── configs/diffusion_v0.yaml
```

The full code for each adapter is in [`docs/research/vla_selection.md`](vla_selection.md).

---

## Shared Obs-Dict Convention

All three adapters must receive observations in this exact format (keys and shapes):

| Key | Shape | dtype | Description |
|---|---|---|---|
| `"state"` | `(16,)` | float32 | Joint positions + gripper states for both arms |
| `"image_head"` | `(224, 224, 3)` | uint8 | Head / overhead camera, RGB |
| `"image_wrist_left"` | `(224, 224, 3)` | uint8 | Left wrist camera, RGB |
| `"image_wrist_right"` | `(224, 224, 3)` | uint8 | Right wrist camera, RGB |

`DummySim` in `contracts/sim.py` currently only emits `"state"` and `"image_head"`.
Rakesh or Akanksh needs to extend `DummySim._obs()` to emit wrist cameras too before
the adapter can be smoke-tested end-to-end.

---

## Action Space Convention

All adapters must return `np.ndarray` of shape `(chunk_size=10, action_dim=16)`.

Expected action vector ordering (**needs confirmation from provided Isaac Sim docs**):
```
index  0– 6: left arm joint positions q_l0 … q_l6  (radians, absolute)
index  7:    left gripper opening width             (metres, 0.0 = closed, 0.08 = open)
index  8–14: right arm joint positions q_r0 … q_r6 (radians, absolute)
index 15:    right gripper opening width            (metres)
```

> ⚠️ **Open Question for Rakesh:** Confirm from the provided Franka FR3 Duo Isaac Sim
> environment whether joint actions are **absolute positions** or **delta displacements**.
> This determines whether the lerobot training config needs `action_space: absolute` or `delta`.
> Record the answer in `configs/robots/fr3_duo.yaml` and `docs/decisions.md`.

---

## Dataset Normalisation Requirement

`lerobot` training requires a `stats.json` file alongside the dataset with:
```json
{
  "action": {"mean": [...16 floats...], "std": [...16 floats...],
              "min": [...16 floats...], "max": [...16 floats...]},
  "observation.state": {"mean": [...16 floats...], "std": [...16 floats...], ...}
}
```

**Rakesh's 01 Ingest script should emit this.** If not yet done, run:
```bash
python -c "from lerobot.scripts.compute_stats import compute_stats; compute_stats('cathlab/handover@v0')"
```
and store the output as `datasets/cathlab_handover_v0/stats.json`.

---

## Unblocking π0.5 (for Rakesh + Sharan)

π0.5 is deferred but Rakesh can pre-wire the adapter shell now. It unblocks when:
1. Sharan ships `serve/client.py::PolicyClient(url: str)` implementing `predict(obs, instruction) -> ndarray`.
2. The openpi server is running on the GPU workstation.

The `Pi05Adapter.load(ckpt)` receives a WebSocket URL as `ckpt`, e.g. `"ws://192.168.x.x:8765"`.

---

## Open Questions Blocking v0

| # | Question | Who answers | Urgency |
|---|---|---|---|
| 1 | Is FR3 Duo action space absolute or delta joints? | Rakesh (check Isaac env docs) | **BLOCKING v0 training** |
| 2 | Does 01 Ingest emit `stats.json` with normalisation stats? | Rakesh | **BLOCKING v0 training** |
| 3 | Does `DummySim` need wrist camera obs for smoke tests? | Akanksh (can fix in eval) | Before H10 |
| 4 | What port/auth does Sharan's `PolicyClient` use? | Sharan | Needed for v1 only |
