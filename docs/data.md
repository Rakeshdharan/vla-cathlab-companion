# Provided Data

Source: the Siemens Healthineers hackathon file share (C2 "Teach the robot"). **Restricted.** These are private HF repos; never make them public or commit them to git.

| HF repo (private, `repo_type=dataset`) | Original zip | Episodes | Frames | FPS | Instruction (single task) |
|---|---|---|---|---|---|
| `Rakeshdharan56/fr3_duo_handover_10ep` | `FR3Duo_Handover_Dataset_lerobot3.zip` | 10 | 22,470 | 30 | "Hand the box from the left gripper to the right gripper and place it on the serving tray." |
| `Rakeshdharan56/fr3_duo_handover_84ep` | `fr3_duo_handover_84_ep.zip` (`v9_night_84`) | 84 | 188,748 | 30 | same as above |
| `Rakeshdharan56/fr3_duo_rand490` | `fr3_duo_rand500.zip` (+ `norm_stats.json`) | 490 | 172,526 | **6** | "stack the three cubes into one tower" |
| `Rakeshdharan56/cathlab_env_isaac` | `Environment_Hackathon_2026.zip` | — | — | — | Provided Isaac Sim / USD scene: **not used** (Isaac dropped at H24); also contains a copy of the 10-episode dataset |

**Common schema (LeRobot v3.0, `robot_type: fr3_duo`):**
- `action` and `observation.state`: `float32[16]` = `left_joint1..7, left_gripper, right_joint1..7, right_gripper`.
- Cameras (video, 224×224×3): `observation.images.external_camera`, `observation.images.left_wrist_camera`, `observation.images.right_wrist_camera`.
- Extra columns: `success: bool`; the handover sets also have `raw_state_row`, `raw_action_row`.

## Access

```bash
hf auth login                      # needs read access to the repos (see note below)
```

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("Rakeshdharan56/fr3_duo_handover_84ep")   # resolves the v3.0 tag
```

```bash
hf download Rakeshdharan56/cathlab_env_isaac --repo-type dataset --local-dir .cache/env
```

> **Teammate access:** private repos under a personal account are visible only to that account. To share them, create a free HF organization, invite the team, and move the repos there (repo **Settings → Rename or transfer**). Then update the IDs in this file.

Re-upload or refresh: `python scripts/upload_provided_data.py --namespace <user-or-org>` (idempotent and resumable).

## Implications for the plan

- The provided demos are a **bimanual box handover** (94 episodes) and **cube stacking** (490 episodes at 6 fps), each with a single fixed instruction. There are no demos of "hand me the `<object>`" with multiple cathlab objects.
- So the language-conditioned cathlab task data must come from the **DataEngine** (Mitesh), and **instruction enrichment** in 01 Ingest is essential.
- The provided sets are best used for: v0 baselines on their own tasks, pretraining/co-training of the FR3 Duo action space, and validating the eval harness.
- Mixing the 6 fps and 30 fps sets needs resampling to a common rate (or training separately). Decide in 01 Ingest.
