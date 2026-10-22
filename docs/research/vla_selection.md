# VLA Selection — CathLab Companion

**Owner:** Akanksh · **Due:** H6 · **Status:** Draft (2026-10-22 H1)

## Task Context

- **Robot:** Franka FR3 Duo (bimanual, 16-DoF: 7 joints + 1 gripper × 2 arms)
- **Task:** Pick-and-place with language grounding ("Hand me the {object}")
- **Training compute:** ≥ 40 GB GPU (A100 / H100)
- **Dataset:** 500 episodes, LeRobot v2/v3 format (parquet + mp4)
- **Framework:** LeRobot-native preferred; openpi adapter acceptable

---

## Evaluation Criteria

| Criterion | Weight | Rationale |
|---|---|---|
| Bimanual support | High | FR3 Duo is 16-DoF; must handle both arms |
| LeRobot / openpi integration | High | Standard training + serving pipeline |
| Fine-tune cost (≥40 GB GPU) | High | A100 budget; need v0 by H10 |
| Inference latency | Medium | p95 ≤ 200 ms budget (`configs/serve.yaml`) |
| Language grounding | High | Core task: pick the *named* object |
| Licence | Medium | Open or permissive for hackathon demo |

---

## Candidate Models

### 1. SmolVLA — HuggingFace / LeRobot

| Criterion | Assessment |
|---|---|
| Bimanual support | ✅ `action_dim` configurable; 16-DoF supported natively in LeRobot |
| LeRobot integration | ✅ Native — no adapter needed |
| Fine-tune cost | ✅ ~2–4 h on A100 for 500 episodes |
| Inference latency | ✅ ~20–50 ms (chunk_size=10, control_hz=30) |
| Language grounding | ✅ Instruction-conditioned via text encoder |
| Licence | Apache-2.0 |
| **Verdict** | **✅ CHOSEN — Fast baseline; all loop iterations** |

**Adapter notes for Rakesh:**
```python
# train/smolvla/adapter.py
from lerobot.policies.smolvla import SmolVLAPolicy
class SmolVLAAdapter(PolicyAdapter):
    def load(self, ckpt): self._policy = SmolVLAPolicy.from_pretrained(ckpt)
    def predict(self, obs, instruction): return self._policy.select_action(obs, instruction)
    def meta(self): return PolicyMeta(name="smolvla", params_m=450, chunk_size=10, control_hz=30, action_dim=16)
```

---

### 2. π0 / π0.5 — Physical Intelligence (openpi)

| Criterion | Assessment |
|---|---|
| Bimanual support | ✅ Designed for bimanual (BridgeV2 + Aloha datasets) |
| LeRobot integration | ⚠️ openpi has LeRobot port; PolicyClient adapter via WebSocket |
| Fine-tune cost | ⚠️ ~8–16 h on A100 (π0); π0.5 larger |
| Inference latency | ⚠️ ~80–150 ms; runs as separate server |
| Language grounding | ✅ Strong — PaliGemma VLM backbone |
| Licence | Apache-2.0 (openpi) |
| **Verdict** | **✅ CHOSEN — Primary frontier candidate (train once on v1 data)** |

**Adapter notes for Rakesh:**
- Deploy `openpi` server on the GPU workstation: `python -m openpi.serve --ckpt <path> --port 8765`
- `PolicyAdapter` → `serve/client.py::PolicyClient("ws://workstation:8765")`
- Action space: configure `action_dim=16` in `openpi` config for FR3 Duo

---

### 3. NVIDIA GR00T N1.x

| Criterion | Assessment |
|---|---|
| Bimanual support | ✅ Designed for humanoid/bimanual (GR1 robot) |
| LeRobot integration | ⚠️ Isaac Lab ecosystem; `gr00t.policy` adapter needed |
| Fine-tune cost | ⚠️ ~10–24 h on A100; N1.5 requires H100 |
| Inference latency | ⚠️ ~100–200 ms |
| Language grounding | ✅ Multimodal foundation model |
| Licence | NVIDIA Open Model Licence (non-commercial) |
| **Verdict** | **✅ CHOSEN (if time) — Secondary frontier; Isaac ecosystem advantage** |

**Adapter notes for Rakesh:**
- `gr00t.policy.load_checkpoint(ckpt_path)` → wrap in `PolicyAdapter`
- Check action space mapping: GR00T uses relative joint targets; FR3 Duo needs absolute or delta

---

### 4. X-VLA

| Criterion | Assessment |
|---|---|
| Bimanual support | ⚠️ Single-arm primary; bimanual via action masking |
| LeRobot integration | ❌ Custom framework; heavy adapter effort |
| Fine-tune cost | ⚠️ Large (7B+ params) |
| Inference latency | ⚠️ ~150 ms |
| Language grounding | ✅ Good zero-shot grounding |
| Licence | Research only (non-commercial) |
| **Verdict** | **○ Documented — evaluate if time allows after H20** |

---

### 5. OpenVLA-OFT

| Criterion | Assessment |
|---|---|
| Bimanual support | ⚠️ Single-arm primary; action dim extension possible |
| LeRobot integration | ⚠️ HF Transformers compatible; medium adapter effort |
| Fine-tune cost | ✅ OFT (orthogonal fine-tuning) is parameter-efficient |
| Inference latency | ⚠️ ~100 ms |
| Language grounding | ✅ OpenVLA-7B backbone; strong zero-shot |
| Licence | Apache-2.0 |
| **Verdict** | **○ Documented — evaluate if time allows after H20** |

---

### 6. ACT (Action Chunking Transformer) — Non-VLA baseline

| Criterion | Assessment |
|---|---|
| Bimanual support | ✅ Designed for Aloha bimanual robot |
| LeRobot integration | ✅ Native |
| Fine-tune cost | ✅ Very fast (~1 h on A100) |
| Inference latency | ✅ ~10–30 ms |
| Language grounding | ❌ No language conditioning; fixed task only |
| Licence | Apache-2.0 |
| **Verdict** | **✅ CHOSEN — Non-VLA lower bound; train alongside v0** |

**Adapter notes for Rakesh:**
```python
# train/act/adapter.py — instruction arg unused
class ACTAdapter(PolicyAdapter):
    def predict(self, obs, instruction): return self._policy.select_action(obs)
    def meta(self): return PolicyMeta(name="act", params_m=86, chunk_size=10, control_hz=30, action_dim=16)
```

---

### 7. Diffusion Policy — Non-VLA baseline (optional)

| Criterion | Assessment |
|---|---|
| Bimanual support | ✅ Arbitrary action dims |
| LeRobot integration | ✅ Native |
| Fine-tune cost | ✅ Fast |
| Inference latency | ⚠️ ~50–100 ms (DDIM steps) |
| Language grounding | ❌ No language |
| Licence | MIT |
| **Verdict** | **○ Optional — if ACT lower bound is ambiguous** |

---

## Final Decision: Shortlist

| Rank | Model | Role | Target Milestone |
|---|---|---|---|
| 1 | **SmolVLA** | Fast baseline + loop iterations v0→v2 | v0 by H10 |
| 2 | **π0.5** | Primary frontier; benchmark headline | v1 by H20 |
| 3 | **ACT** | Non-VLA lower bound | v0 by H10 |
| 4 | **GR00T N1.x** | Secondary frontier (if time ≥ H20) | v1 by H20 |

---

## Comparison Matrix (to be filled during benchmark A5)

| Model | Bimanual | LeRobot | Fine-tune h (A100) | Latency p95 ms | Licence | Status |
|---|---|---|---|---|---|---|
| SmolVLA | ✅ | ✅ Native | ~2–4 | ~50 | Apache-2.0 | ✅ Primary |
| π0.5 | ✅ | ⚠️ Adapter | ~8–16 | ~150 | Apache-2.0 | ✅ Primary |
| GR00T N1.x | ✅ | ⚠️ Adapter | ~10–24 | ~200 | NVIDIA OML | ✅ If time |
| ACT | ✅ | ✅ Native | ~1 | ~30 | Apache-2.0 | ✅ Baseline |
| X-VLA | ⚠️ | ❌ | large | ~150 | Research | ○ Documented |
| OpenVLA-OFT | ⚠️ | ⚠️ | medium | ~100 | Apache-2.0 | ○ Documented |
| Diffusion Policy | ✅ | ✅ Native | ~1 | ~100 | MIT | ○ Optional |

---

## Notes

- All bimanual adapters: set `action_dim=16` (7 joints + 1 gripper × 2 arms).
- If a model only supports `action_dim ≤ 7`, use the adapter to mask the inactive arm and record in its `ModelCard`.
- Latency numbers are estimates from literature; actual p95 will be measured by the eval harness (`eval/cli.py --vlm-judge`).
- Record final choice in `docs/decisions.md` at H6.
