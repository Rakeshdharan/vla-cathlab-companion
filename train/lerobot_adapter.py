"""One PolicyAdapter for every LeRobot checkpoint we train (ACT, SmolVLA, X-VLA): `lerobot:<repo-or-path>`.

The policy class comes from the checkpoint's own config, and the saved pre/post-processors do the camera rename,
tokenisation and (un)normalisation exactly as in training, so this adapter only feeds the *dataset* keys:
`observation.state`, `observation.images.<camera>` (CHW float in [0, 1]) and `task`. lerobot/torch import lazily.
"""
from typing import Callable
import numpy as np
from contracts import PolicyAdapter, PolicyMeta

STATE_KEY = "state"


def to_frame(obs: dict[str, np.ndarray], instruction: str) -> dict:
    """Our obs dict -> a batch of one in LeRobot dataset keys (numpy; images HWC uint8 -> CHW float32 in [0, 1])."""
    frame: dict = {"task": [instruction]}
    for key, value in obs.items():
        value = np.asarray(value)
        if key == STATE_KEY:
            frame["observation.state"] = value.astype(np.float32)[None]
        elif value.ndim == 3 and value.shape[-1] == 3:
            frame[f"observation.images.{key}"] = (value.transpose(2, 0, 1).astype(np.float32) / 255.0)[None]
    return frame


def to_actions(chunk: np.ndarray, n_action_steps: int, action_dim: int) -> np.ndarray:
    """(1, T, D) or (T, D) postprocessed chunk -> (min(T, n_action_steps), action_dim) float32 in [-1, 1]."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if chunk.ndim == 3:
        chunk = chunk[0]
    return np.clip(chunk[:n_action_steps, :action_dim], -1.0, 1.0)


class LeRobotPolicyAdapter(PolicyAdapter):
    def __init__(self, ckpt: str | None = None, device: str | None = None, revision: str | None = None,
                 action_dim: int = 7, control_hz: float = 20.0, n_action_steps: int | None = None):
        self.device, self.revision = device, revision
        self.action_dim, self.control_hz = action_dim, control_hz
        self.n_action_steps = n_action_steps
        self._infer: Callable[[dict], np.ndarray] | None = None
        self._meta = PolicyMeta(name="lerobot", params_m=0.0, chunk_size=n_action_steps or 1, control_hz=control_hz,
                                action_dim=action_dim)
        if ckpt:
            self.load(ckpt)

    def load(self, ckpt: str) -> None:
        """Load a checkpoint. `load("")` after a load is a no-op (engines call it on factory-built policies)."""
        if not ckpt:
            if self._infer is None:
                raise ValueError("LeRobotPolicyAdapter.load needs a checkpoint path or HF repo id")
            return
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        cfg = PreTrainedConfig.from_pretrained(ckpt, revision=self.revision)
        cfg.device = device
        policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg, revision=self.revision)
        policy.to(device).eval()
        pre, post = make_pre_post_processors(cfg, pretrained_path=ckpt, pretrained_revision=self.revision,
                                             preprocessor_overrides={"device_processor": {"device": device}})
        n_steps = self.n_action_steps or int(getattr(cfg, "n_action_steps", None) or getattr(cfg, "chunk_size", 1))
        self.n_action_steps = n_steps

        def infer(frame: dict) -> np.ndarray:
            batch = {k: (v if k == "task" else torch.from_numpy(v)) for k, v in frame.items()}
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(pre(batch)))
            return chunk.float().cpu().numpy()

        self._infer = infer
        params_m = sum(p.numel() for p in policy.parameters()) / 1e6
        self._meta = PolicyMeta(name=f"{cfg.type}:{ckpt}", params_m=round(params_m, 1), chunk_size=n_steps,
                                control_hz=self.control_hz, action_dim=self.action_dim)

    def predict(self, obs: dict[str, np.ndarray], instruction: str) -> np.ndarray:
        if self._infer is None:
            raise RuntimeError("LeRobotPolicyAdapter.load(ckpt) must be called first")
        return to_actions(self._infer(to_frame(obs, instruction)), self.n_action_steps or 1, self.action_dim)

    def meta(self) -> PolicyMeta:
        return self._meta
