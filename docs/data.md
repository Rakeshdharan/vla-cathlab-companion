# Data

Two kinds of datasets live on the Hub: the hackathon's **provided** data (restricted, FR3 Duo), and the datasets we
**generate** ourselves with the scripted expert in each engine (public, Panda + FR3). All are LeRobot v3.

## Provided data (FR3 Duo, restricted)

Source: the Siemens Healthineers hackathon file share (C2 "Teach the robot"). **Restricted.** These are private HF
repos; never make them public or commit them to git.

| HF repo (private, `repo_type=dataset`) | Episodes | Frames | FPS | Instruction (single task) |
|---|---|---|---|---|
| `Rakeshdharan56/fr3_duo_handover_10ep` | 10 | 22,470 | 30 | "Hand the box from the left gripper to the right gripper and place it on the serving tray." |
| `Rakeshdharan56/fr3_duo_handover_84ep` | 84 | 188,748 | 30 | same as above |
| `Rakeshdharan56/fr3_duo_rand490` | 490 | 172,526 | 6 | "stack the three cubes into one tower" |

Common schema (`robot_type: fr3_duo`): `action` / `observation.state` are `float32[16]` =
`left_joint1..7, left_gripper, right_joint1..7, right_gripper`; cameras (video, 224×224×3):
`observation.images.{external_camera,left_wrist_camera,right_wrist_camera}`; extra columns `success: bool`
(handover sets also have `raw_state_row`, `raw_action_row`).

Access:
```bash
hf auth login   # needs read access to the repos
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; LeRobotDataset('Rakeshdharan56/fr3_duo_handover_84ep')"
```

These demos are a bimanual box handover and cube stacking, each with a single fixed instruction — there is no
"hand me the `<object>`" data with multiple cathlab objects in the provided set, so the language-conditioned cathlab
task data comes from our own **DataEngine** below.

## Generated data (our pick task)

[`configs/datasets.yaml`](../configs/datasets.yaml) is the source of truth the DATA tab reads
(`GET /api/datasets`); update it when a dataset lands.

| Dataset | Robot / I/O | Engine (physics) | Episodes | Frames | FPS | Notes |
|---|---|---|---|---|---|---|
| [`Rakeshdharan56/pick_robosuite`](https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite) `v1` | Panda, `panda_libero` | robosuite 1.5.2 (MuJoCo 3.3.7) | 384 | 69,191 | 20 | Full `cathlab_v1` train grid, expert 100 % |
| [`Rakeshdharan56/pick_maniskill`](https://huggingface.co/datasets/Rakeshdharan56/pick_maniskill) `v1` | Panda, `panda_libero` | ManiSkill3 3.0.1 (PhysX 5) | 369 | 45,387 | 20 | Paired with robosuite episode for episode |
| [`Rakeshdharan56/pick_genesis`](https://huggingface.co/datasets/Rakeshdharan56/pick_genesis) `v1` | Panda, `panda_libero` | Genesis 1.4.5 | 383 | 66,447 | 20 | Paired with robosuite, plus Genesis-only camera jitter / light direction |
| [`Rakeshdharan56/pick_panda_v2`](https://huggingface.co/datasets/Rakeshdharan56/pick_panda_v2) `v1` | Panda, `panda_libero` | robosuite + ManiSkill + Genesis (merged) | 1,136 | 181,025 | 20 | Merge of the three v1 sets, for cross-sim co-training |
| [`aravindsuresh16/cathlab_fr3_newton_v1`](https://huggingface.co/datasets/aravindsuresh16/cathlab_fr3_newton_v1) `v1.0.1` | FR3 Duo, `fr3_duo_16d` | Newton 1.6 (MuJoCo-Warp) | 384 | 159,568 | 30 | Use `v1.0.1` (widened q01/q99 of the idle left arm) |

**I/O profiles** (`engines.registry.IO_PROFILES`):
- `panda_libero`: 7-D normalized EEF-delta action, 8-D state, `agentview` + `eye_in_hand` 256×256.
- `fr3_duo_16d`: 16-D joint targets (both arms + grippers), 3 cameras 224×224.

## Access and re-upload

```bash
python -m engines.robosuite.cli convert --dataset <generate --out> --out <dir>   # engine-generated: npz -> LeRobot v3
```

Re-uploading or refreshing the provided data is occasional hackathon-admin tooling, not part of the product; the
script for it lives on the `pre-dev` branch if needed again.

> **Private-repo access:** repos under a personal HF account are visible only to that account. To share them,
> create a free HF organization, invite the team, and transfer the repos (**Settings → Rename or transfer**), then
> update the IDs here and in `configs/datasets.yaml`.

## P6 data-quality rules (`dataengine/quality.py`, `dataengine/generate.py`)

Every engine's generator and converter apply the same rules before an episode is written:

- **No-op frames dropped:** LIBERO's rule — no translation/rotation and the gripper command unchanged from the
  previous action (`dataengine.quality.is_noop`).
- **Success must hold:** an episode is only kept successful if the task stays succeeded for
  `ExpertParams.success_hold` (10) consecutive steps up to the end, not just the first instant it's true.
- **Shard count checks:** `dataengine.npz_sink.validate_shard` checks the manifest's episode count, the
  `episodes_extra.jsonl` line count and the `.npz` file count agree, and each episode's frame count matches its
  action array, before shards are merged (`merge_raw_shards` raises on any mismatch).
- **Provenance per episode:** `dataengine.quality.base_provenance` / `engine_provenance` record the git commit
  (+ dirty flag for the data code), Python version and engine/library versions with every generated episode.

## Dataset intelligence

```bash
uv run --with pyarrow python -m dataset_intel.report <lerobot_root> [--extra <root>/meta/episodes_extra.jsonl] \
    [--io panda_libero|fr3_duo_16d] [--spec configs/variation/cathlab_v1.yaml] --out var/intel/<name>
```

Validators, train-grid coverage and a readiness score for any LeRobot v3 dataset, without a GPU. See
[dataset_intel/README.md](../dataset_intel/README.md).
