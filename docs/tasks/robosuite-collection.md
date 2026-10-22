# Collecting robosuite episodes (teammate guide)

You will produce demonstrations of a Franka Panda robot doing *"Hand me the {bottle | vial | syringe}"*: it picks the named object from a table and puts it in a tray. A scripted "expert" controller does the robot work. You only run commands, check the numbers, and upload the result. No robotics knowledge is needed.

**What you produce:** a LeRobot v3 dataset (`cathlab/pick_robosuite`, version `v1`) with 384 episodes: 3 objects × 8 layouts × 4 object colours × 4 table colours. Each episode has 2 camera videos (256×256), an 8-D robot state and a 7-D action at 20 Hz.

**Rule:** data never goes into git. `data/` and `outputs/` are git-ignored; finished datasets go to the private Hugging Face org.

## 1. One-time setup

You need two Python environments, because LeRobot needs Python ≥ 3.12 and robosuite runs on 3.10.

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks

# robosuite env: generates episodes
conda create -y -n robosuite python=3.10 && conda activate robosuite
pip install robosuite==1.5.2 "mujoco==3.3.7" "numpy<2" pydantic pyyaml imageio pytest
conda deactivate

# lerobot env: converts and uploads (skip if you already have one with lerobot 0.5.x)
conda create -y -n lerobot python=3.12 && conda activate lerobot
pip install "lerobot>=0.5,<0.6" pytest
conda deactivate
```

> **Keep `mujoco==3.3.7` and `numpy<2`** (the verified versions). With mujoco 3.16, robosuite 1.5.2 crashes on reset with `AssertionError` in `get_joint_qpos_addr`.

Check that everything works (about 40 s; all tests should pass):

```bash
conda activate robosuite
export MUJOCO_GL=egl          # off-screen rendering on the GPU; use MUJOCO_GL=osmesa if EGL is missing
python -m pytest -q tests/sim
```

All commands below run from the repo root with `MUJOCO_GL=egl` set.

## 2. Generate your shard

The 384 configurations are split into **shards** so several processes or people can run in parallel. Everyone uses the same `--n` and `--num-shards`, and each process gets a different `--shard` (0 … k−1). Ask Rakesh which shard numbers are yours.

```bash
conda activate robosuite
export MUJOCO_GL=egl
# example: you run shard 2 of 8
python scripts/robosuite_generate.py --n 384 --num-shards 8 --shard 2 --out data/pick_robosuite --created-by "<your name>"
```

To run several shards on one machine, start one process per shard in separate terminals, or:

```bash
for i in 0 1 2 3; do
  python scripts/robosuite_generate.py --n 384 --num-shards 8 --shard $i --out data/pick_robosuite --created-by "<your name>" > gen_$i.log 2>&1 &
done; wait
```

- Each process does about **3 episodes/min**. 48 configurations (one shard of 8) take about 15–20 min. One process uses about 1 CPU core and 1 GB RAM.
- Output: `data/pick_robosuite/raw/shard_002-of-008/` containing `episodes/*.npz` (5–11 MB each), `meta/episodes_extra.jsonl` and `manifest.yaml`.
- A shard folder can't be reused. To redo a shard, delete its folder first.

## 3. Check the success rate and look at the episodes

The last lines of each run print something like:

```
kept 48/48 (100%); per object {'bottle': '16/16', 'vial': '16/16', 'syringe': '16/16'}
2.9 episodes/min attempted (16.6 min)
```

- **Expected:** 100%, or very close to it. Only successful episodes are kept, so a failure just means one config is missing from the dataset.
- **Below 80% for any object:** stop and tell Rakesh. It means something changed in the environment (package versions, scene, expert parameters).

Watch a few episodes (left: `agentview` camera, right: wrist camera):

```bash
python scripts/preview_episodes.py data/pick_robosuite --episodes 0 1 2 --out outputs/preview.gif
```

To test the expert without recording anything (faster, no images), run 20 configurations per object:

```bash
python scripts/robosuite_expert_check.py --n 20
```

### Reading a failure

`robosuite_expert_check.py` prints one line per failed episode:

```
FAIL vial layout=8 seed=10000 steps=300 trace=['approach', 'pre_grasp'] flags={'reach': False, ...}
```

- `trace` is the list of waypoints the robot reached, in order: `rise` (only if it starts low), `approach`, `pre_grasp`, `grasp`, `close`, `lift`, `above_tray`, `lower`, `open`, `retreat`. The failure happened right after the last one listed.
- `flags` are the task stages: `reach`, `grasp`, `lift`, `transport`, `place`. The first `False` is the stage that failed.
- `steps=300` means the episode ran out of time (the robot was stuck).

Send Rakesh the command you ran, the `FAIL` lines and the preview GIF.

## 4. Merge all shards into one LeRobot dataset

Once all shards are finished, copy every `raw/shard_*` folder into one machine's `data/pick_robosuite/raw/` (for example with `rsync` or `hf download`, see step 5). Then convert:

```bash
conda activate lerobot
python scripts/npz_to_lerobot.py --dataset data/pick_robosuite
# -> data/pick_robosuite/lerobot/  (videos, parquet, meta/episodes_extra.jsonl, manifest.yaml)
```

Check that it loads:

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('cathlab/pick_robosuite', root='data/pick_robosuite/lerobot')
f = ds[0]
print(ds.num_episodes, 'episodes', ds.fps, 'fps', f['action'].shape, f['observation.state'].shape, f['observation.images.agentview'].shape)"
# expected: 384 episodes 20 fps torch.Size([7]) torch.Size([8]) torch.Size([3, 256, 256])
```

## 5. Upload to Hugging Face as `cathlab/pick_robosuite@v1`

The repo is **private** and must stay private.

```bash
conda activate lerobot
hf auth login                                     # once; needs write access to the cathlab org
hf upload cathlab/pick_robosuite data/pick_robosuite/lerobot . --repo-type dataset --private \
  --commit-message "pick_robosuite v1: 384 scripted robosuite episodes"
hf repos tag create cathlab/pick_robosuite v1 --repo-type dataset -m "robosuite scripted, train grid 384"
```

Optional: also keep the raw shards, so the dataset can be rebuilt without re-simulating:

```bash
hf upload cathlab/pick_robosuite_raw data/pick_robosuite/raw . --repo-type dataset --private
```

Then post the dataset link and the per-object success numbers in the team channel.

## Who to ask

| Problem | Ask |
|---|---|
| Install, robosuite/mujoco errors, success < 80%, scene or expert questions | Rakesh |
| Pairing with the Isaac episodes (same configs, same camera) | Mitesh |
| Using the dataset for evaluation | Akanksh |
| HF org access | Rakesh |

## Reference

- Scene, adapter, randomisation: [`sim/robosuite/`](../../sim/robosuite/README.md)
- Variation grid and expert core: [`configs/variation/cathlab_v1.yaml`](../../configs/variation/cathlab_v1.yaml), [`dataengine/`](../../dataengine/)
- Why things are the way they are: [decisions log](../decisions.md)
