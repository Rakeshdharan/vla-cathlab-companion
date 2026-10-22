# VLA Selection — CathLab Companion (Deliverable A1)

**Owner:** Akanksh · **Due:** H6 · **Status:** ✅ LOCKED (2026-10-22 H6)  
**Reviewed by:** Akanksh · **Pending Rakesh sign-off on adapter interfaces**

---

## Decision Summary (Lock-In)

> **LeRobot-native-first constraint:** Given the 36-hour hackathon timeline and a dataset already in
> LeRobot v2/v3 format, every model that requires a non-trivial framework adapter is deferred until
> after v0 is demonstrated. Only models with native `lerobot` policy classes are cleared for
> immediate fine-tuning and evaluation.

| Role | Model | LeRobot Status | Decision |
|---|---|---|---|
| **VLA Baseline** | SmolVLA | ✅ Native | **LOCKED — train v0** |
| **Non-VLA Lower Bound** | ACT | ✅ Native | **LOCKED — train v0** |
| **Non-VLA Optional** | Diffusion Policy | ✅ Native | Activate if ACT result is ambiguous |
| **Frontier VLA (later)** | π0.5 (openpi) | ⚠️ Port exists, needs adapter | Deferred to v1 (H20) |
| **Frontier VLA (stretch)** | GR00T N1.x | ⚠️ Isaac Lab adapter needed | Deferred, if time ≥ H20 |
| **Documented only** | OpenVLA-OFT | ⚠️ HF Transformers, not native | Out of scope for hackathon |
| **Documented only** | X-VLA | ❌ Custom framework | Out of scope |

---

## Task Context

| Item | Value |
|---|---|
| Robot | Franka FR3 Duo — bimanual, 16-DoF (7 joints + 1 gripper) × 2 arms |
| Action space | `action_dim = 16`, continuous, chunk-based at 30 Hz |
| Observation space | Proprioception (16D state) + 3 cameras (head, wrist_left, wrist_right) at 224×224 |
| Dataset | 500 episodes, LeRobot v2/v3 format (parquet + mp4 + `manifest.yaml`) |
| Training compute | ≥ 40 GB GPU (A100 or H100) on GPU workstation |
| Inference budget | p95 latency ≤ 200 ms (`configs/serve.yaml`) |
| Language task | `"Hand me the {object}"` — 6 cathlab objects |
| Framework constraint | **LeRobot-native for v0** |

---

## What "LeRobot Native" Means

A model is **LeRobot native** when all of the following are true:

1. The policy class lives inside the `lerobot` Python package (e.g., `lerobot.policies.smolvla`).
2. It is trained via `lerobot`'s unified training CLI without a third-party server or framework.
3. It loads a checkpoint and runs inference with the standard `policy.select_action(obs_dict)` or
   `policy.select_action(obs_dict, task)` interface.
4. It reads data directly from the LeRobot dataset format (Parquet + HF Hub) with no conversion step.

Models meeting all four criteria: **SmolVLA**, **ACT**, **Diffusion Policy**.

---

## Locked-In Models (v0 — Train Immediately)

### 1. SmolVLA

**Role:** Primary VLA baseline. Drives v0 → v1 → v2 loop iterations.

#### What It Is
SmolVLA is a compact Vision-Language-Action model developed natively inside the `lerobot` library.
It couples a lightweight SmolVLM-style vision-language backbone (compressed vision encoder + small
language model) to a continuous action-chunking head. It does not use autoregressive action
tokenisation — instead it predicts a full chunk of `chunk_size` future actions in one forward pass.

#### Architecture
```
[Head Camera Image  (224×224×3)] ──► [Vision Encoder (ViT-S)] ──┐
[Wrist Left Image   (224×224×3)] ──► [Vision Encoder (ViT-S)] ──┤
[Wrist Right Image  (224×224×3)] ──► [Vision Encoder (ViT-S)] ──┤─► [Cross-Attention Fusion]
                                                                  │           │
[Instruction: "Hand me the guidewire"] ──► [Text Tokenizer] ──────┘           ▼
[Proprioception (16D state)]          ────────────────────────────► [Action Head (MLP)]
                                                                              │
                                                                              ▼
                                                               Action Chunk: (10 × 16)
                                                               = 10 future steps × 16 DoF
```

#### Why It Wins for v0
- `lerobot.policies.smolvla.SmolVLAPolicy` exists as a first-class class — no adapter code required
  beyond wrapping `select_action()` in the `PolicyAdapter` interface.
- Trains in **2–4 hours** on a single A100 with 500 episodes, meaning v0 can be demonstrated by H10.
- **Instruction-conditioned** natively: the `task` argument to `select_action(obs, task)` passes the
  natural language instruction directly into the text encoder. This is the key VLA requirement —
  the model picks different objects based on language.
- Inference: single forward pass at ~20–40 ms, well under the 200 ms p95 budget.
- Apache-2.0 licence.

#### Fine-Tuning Configuration
```yaml
# configs/models/smolvla_v0.yaml  (to be created by Rakesh in train/)
policy: smolvla
pretrained_name_or_path: lerobot/smolvla_base   # HF Hub ID
dataset_repo_id: cathlab/handover@v0
batch_size: 8
steps: 50000
lr: 1e-4
action_dim: 16
chunk_size: 10
control_hz: 30
cameras: [head, wrist_left, wrist_right]
image_size: [224, 224]
```

#### Concrete PolicyAdapter Interface for Rakesh
```python
# train/smolvla/adapter.py
from __future__ import annotations
import numpy as np
from contracts import PolicyAdapter, PolicyMeta

class SmolVLAAdapter(PolicyAdapter):
    """Wraps lerobot SmolVLAPolicy in the shared PolicyAdapter contract."""

    def load(self, ckpt: str) -> None:
        from lerobot.policies.smolvla import SmolVLAPolicy
        self._policy = SmolVLAPolicy.from_pretrained(ckpt)
        self._policy.eval()

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        # obs keys expected: "state" (16,), "image_head" (224,224,3),
        #                    "image_wrist_left" (224,224,3), "image_wrist_right" (224,224,3)
        import torch
        obs_torch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            chunk = self._policy.select_action(obs_torch, task=instruction)
        return chunk.squeeze(0).cpu().numpy()   # shape: (chunk_size=10, action_dim=16)

    def meta(self) -> PolicyMeta:
        return PolicyMeta(
            name="smolvla",
            params_m=450.0,      # ~450M parameters
            chunk_size=10,
            control_hz=30.0,
            action_dim=16,
        )
```

---

### 2. ACT (Action Chunking Transformer)

**Role:** Non-VLA lower bound. Establishes what pure behavioural cloning without language
grounding achieves. Every VLA must beat ACT to justify its added complexity.

#### What It Is
ACT is a Conditional Variational Autoencoder (CVAE) parameterised by a Transformer encoder-decoder.
It was originally designed for the ALOHA bimanual robot (7+7 DoF), making it directly applicable
to the FR3 Duo.

#### Architecture
```
[All Camera Images (stacked)] ──► [ResNet-18 Backbone] ──► [Visual Tokens]
[Proprioception (16D)]        ──► [Linear Projection]  ──► [State Token]
[Action sequence from demo]   ──► [CVAE Encoder]        ──► [Style Latent z]
                                                                   │
                         ┌─────────────────────────────────────────┘
                         ▼
                 [Transformer Decoder]
                 (visual tokens + state token + latent z)
                         │
                         ▼
              Action Chunk: (chunk_size × 16)   ← predicts H steps ahead

At inference: z ~ N(0,I), no encoder needed.
```

**Temporal Ensembling:** ACT runs inference every step but uses an exponentially weighted average
of overlapping action chunk predictions to smooth trajectories:
$$a_t = \frac{\sum_{i} w_i \hat{a}_{t,i}}{\sum_i w_i}, \quad w_i = e^{-k \cdot i}$$

#### Key Distinction from SmolVLA
ACT takes **no language input**. The `instruction` argument to `PolicyAdapter.predict()` is
received but **ignored**. This means:
- ACT cannot distinguish `"hand me the guidewire"` from `"hand me the catheter"`.
- It implicitly assumes the task is always the same pick-and-place.
- This makes it the ideal lower bound: it proves how far pure trajectory imitation gets without
  semantic grounding.

#### Fine-Tuning Configuration
```yaml
# configs/models/act_v0.yaml
policy: act
dataset_repo_id: cathlab/handover@v0
batch_size: 8
steps: 50000
lr: 1e-5
action_dim: 16
chunk_size: 10
control_hz: 30
cameras: [head, wrist_left, wrist_right]
image_size: [224, 224]
kl_weight: 10   # CVAE regularisation weight
```

#### Concrete PolicyAdapter Interface for Rakesh
```python
# train/act/adapter.py
from __future__ import annotations
import numpy as np
from contracts import PolicyAdapter, PolicyMeta

class ACTAdapter(PolicyAdapter):
    """Wraps lerobot ACTPolicy. NOTE: instruction arg is ignored — ACT has no language head."""

    def load(self, ckpt: str) -> None:
        from lerobot.policies.act import ACTPolicy
        self._policy = ACTPolicy.from_pretrained(ckpt)
        self._policy.eval()

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        # instruction is deliberately ignored — ACT is language-free
        import torch
        obs_torch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            chunk = self._policy.select_action(obs_torch)
        return chunk.squeeze(0).cpu().numpy()   # shape: (chunk_size=10, action_dim=16)

    def meta(self) -> PolicyMeta:
        return PolicyMeta(
            name="act",
            params_m=86.0,       # ~86M parameters
            chunk_size=10,
            control_hz=30.0,
            action_dim=16,
        )
```

---

### 3. Diffusion Policy (LeRobot Native — Activate if Needed)

**Role:** Optional second non-VLA baseline. Activates if ACT shows high variance or bimodal
failure modes that a unimodal regression baseline cannot diagnose.

#### What It Is
Diffusion Policy frames robot control as score-matching / DDPM: the policy learns to denoise
a random action sequence back into a clean trajectory conditioned on current observations.
Unlike ACT's CVAE (which collapses multi-modal demonstrations into a single Gaussian latent),
Diffusion Policy can model **multiple valid trajectories** for the same observation.

#### When to Activate
- If ACT success rate on heavy-clutter tier < 30% with high variance across seeds: Diffusion
  Policy's multi-modal distribution may be capturing the demonstration spread more faithfully.
- If the stage failure histogram shows ACT collapsing to a single grasp strategy that
  occasionally catastrophically fails: Diffusion Policy may generalise better.

#### Concrete PolicyAdapter Interface for Rakesh
```python
# train/diffusion/adapter.py
from __future__ import annotations
import numpy as np
from contracts import PolicyAdapter, PolicyMeta

class DiffusionPolicyAdapter(PolicyAdapter):
    """Wraps lerobot DiffusionPolicy. NOTE: instruction arg ignored — no language head."""

    def load(self, ckpt: str) -> None:
        from lerobot.policies.diffusion import DiffusionPolicy
        self._policy = DiffusionPolicy.from_pretrained(ckpt)
        self._policy.eval()

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        import torch
        obs_torch = {k: torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
        with torch.no_grad():
            chunk = self._policy.select_action(obs_torch)
        return chunk.squeeze(0).cpu().numpy()   # shape: (chunk_size=10, action_dim=16)

    def meta(self) -> PolicyMeta:
        return PolicyMeta(
            name="diffusion_policy",
            params_m=321.0,
            chunk_size=10,
            control_hz=30.0,
            action_dim=16,
        )
```

---

## Deferred Models (v1 / Stretch — Adapter Required)

These models are documented so Rakesh can build adapters independently in `train/`. They do not
block v0. They are evaluated in the benchmark A5 at H20 if adapters are ready.

### π0.5 (Physical Intelligence / openpi)

**What blocks it from v0:** π0.5 does not live inside the `lerobot` package. It runs via the
`openpi` framework as a separate inference server (WebSocket). Integrating it requires:
1. A running `openpi` server process on the GPU workstation (separate from the main Python env).
2. Sharan's `PolicyClient` in `serve/client.py` implementing the WebSocket handshake.
3. Configuring `openpi` action space for 16-DoF (`action_dim=16` in the openpi server config).

**Why it's worth it for v1:** π0.5 uses the PaliGemma 3B VLM backbone with flow-matching action
generation. It has natively trained on ALOHA bimanual datasets, making it the strongest available
prior for bimanual pick-and-place with language grounding.

**Deferred interface for Rakesh (train/pi0/adapter.py):**
```python
# Requires: serve/client.py::PolicyClient, openpi server running at ws://workstation:8765
class Pi05Adapter(PolicyAdapter):
    def load(self, ckpt: str) -> None:
        from serve.client import PolicyClient
        self._client = PolicyClient(ckpt)   # ckpt = ws://workstation:8765

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        return self._client.predict(obs, instruction)   # remote call

    def meta(self) -> PolicyMeta:
        return PolicyMeta(name="pi05", params_m=3000.0, chunk_size=10, control_hz=30, action_dim=16)
```

**Dependency:** Sharan must merge `serve/client.py` before this unblocks.

---

### GR00T N1.x (NVIDIA)

**What blocks it from v0:** GR00T N1.x checkpoint weights are distributed via NGC and require
the `gr00t` Python package (Isaac Lab ecosystem). It does not integrate with LeRobot training CLI.
Additionally, NVIDIA's Open Model Licence restricts commercial use — acceptable for hackathon demo
but must be stated explicitly in the `ModelCard`.

**Deferred interface for Rakesh (train/groot/adapter.py):**
```python
# Requires: gr00t package, Isaac Lab env, NVIDIA checkpoint download
class GR00TAdapter(PolicyAdapter):
    def load(self, ckpt: str) -> None:
        from gr00t.policy import GR00TPolicy
        self._policy = GR00TPolicy.from_pretrained(ckpt)
        self._policy.eval()

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        # NOTE: GR00T action space uses relative joint deltas, not absolute.
        # Must convert to absolute positions before returning.
        ...

    def meta(self) -> PolicyMeta:
        return PolicyMeta(name="groot_n1", params_m=1500.0, chunk_size=16, control_hz=30, action_dim=16)
```

**Dependency:** Rakesh must confirm action-space mapping (absolute vs. delta joints) from provided docs.

---

## Final Locked Shortlist

```
v0 (train now, evaluate by H10):
  ├── SmolVLA        ← VLA baseline, LeRobot native, language-grounded
  └── ACT            ← Non-VLA lower bound, LeRobot native, language-free

v0 optional (activate if ACT result is inconclusive):
  └── Diffusion Policy  ← Non-VLA, LeRobot native, multi-modal distribution

v1 (deferred, adapters being built in parallel by Rakesh / Sharan):
  └── π0.5           ← Primary frontier VLA, openpi WebSocket adapter

stretch (if time after H20):
  └── GR00T N1.x     ← Secondary frontier, Isaac Lab adapter
```

---

## Benchmark Comparison Matrix

*Success rates to be filled in by eval/cli.py runs. CI bounds computed via Wilson score interval.*

| Model | LeRobot | Language | Bimanual | Fine-tune h | Latency p95 ms | Licence | v0 Status |
|---|---|---|---|---|---|---|---|
| SmolVLA | ✅ Native | ✅ Yes | ✅ 16-DoF | ~2–4 | ~40 | Apache-2.0 | **Train now** |
| ACT | ✅ Native | ❌ No | ✅ 16-DoF | ~1 | ~25 | Apache-2.0 | **Train now** |
| Diffusion Policy | ✅ Native | ❌ No | ✅ 16-DoF | ~1 | ~80 | MIT | If needed |
| π0.5 | ⚠️ Port | ✅ Yes | ✅ Yes | ~8–16 | ~150 | Apache-2.0 | v1 (H20) |
| GR00T N1.x | ⚠️ Adapter | ✅ Yes | ✅ Yes | ~10–24 | ~200 | NVIDIA OML | stretch |
| OpenVLA-OFT | ❌ | ✅ Yes | ⚠️ Partial | large | ~100 | Apache-2.0 | out of scope |
| X-VLA | ❌ | ✅ Yes | ⚠️ Partial | large | ~150 | Research | out of scope |

---

## Open Questions (for Rakesh)

- [ ] **`action_dim` confirmation:** Verify from provided Isaac Sim env docs that the FR3 Duo
  action vector is exactly 16-D and the ordering is `[q_l0..q_l6, grip_l, q_r0..q_r6, grip_r]`.
  Needed before `lerobot` training config is finalised.
- [ ] **Dataset normalization stats:** `lerobot` training requires pre-computed `stats.json`
  (action mean/std/min/max). Confirm 01 Ingest produces this for the 500-episode dataset.
- [ ] **π0.5 server spec:** What port and authentication does Sharan's `PolicyClient` expect?
  Needed before the pi05 adapter can be written.
