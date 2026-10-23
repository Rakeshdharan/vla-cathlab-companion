# train

| Module | What |
|---|---|
| `lerobot_adapter.py` | `LeRobotPolicyAdapter(PolicyAdapter)`: loads any LeRobot checkpoint (ACT, SmolVLA, X-VLA; HF repo id or path) with its saved pre/post-processors. Engines use it as `--policy lerobot:<repo-or-path>` |

The adapter feeds the dataset keys (`observation.state`, `observation.images.{agentview,eye_in_hand}` CHW in [0, 1], `task`). The checkpoint's preprocessor applies the training camera rename, so there is no per-model mapping here. It returns `(n_action_steps, 7)` float32 clipped to [-1, 1], which the engines execute open-loop.

Needs `lerobot` (and torch) in the env, e.g. `/media/data1/waifzi2/vla-eval` (LeRobot + robosuite + ManiSkill):

```bash
MUJOCO_GL=egl python -m engines.robosuite.cli rollout --policy lerobot:Rakeshdharan56/act_panda_pick_v1 --out var/smoke/act_rollout
python -m engines.maniskill.cli eval --policy lerobot:Rakeshdharan56/smolvla_panda_pick_v1 --out var/eval/maniskill/smolvla_panda_pick_v1
```

Training recipes: [docs/handoffs/2026-10-23-maniskill-training-session.md](../docs/handoffs/2026-10-23-maniskill-training-session.md) §5.
