"""Policy factory for engine eval/rollout: --policy random | url:<policy-server-url> (expert: engine-specific)."""
from contracts import PolicyAdapter, RandomPolicy


def make_policy(spec: str, action_dim: int, seed: int = 0) -> PolicyAdapter:
    """`expert` needs simulator state, so each engine builds engines.expert_policy.ExpertPolicy itself."""
    if spec == "random":
        return RandomPolicy(action_dim=action_dim, chunk_size=1, seed=seed)
    if spec.startswith("url:"):
        try:
            from serve.policy_client import PolicyClient
        except ImportError as e:
            raise RuntimeError("policy server client not available yet (serve/policy_client.py, Sharan's S2)") from e
        return PolicyClient(spec[len("url:"):])
    if spec.startswith("lerobot:"):
        from train.lerobot_adapter import LeRobotPolicyAdapter   # lazy: needs lerobot + torch
        return LeRobotPolicyAdapter(ckpt=spec[len("lerobot:"):], action_dim=action_dim)
    raise ValueError(f"unknown --policy '{spec}': use expert, random, url:<policy-server-url> or "
                     "lerobot:<hf-repo-or-path>")
