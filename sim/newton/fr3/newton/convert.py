# Vendored from aravinds1610/mujoco_cathlab_vla@fc019f8 (cathlab_sim/newton/convert.py).
"""Convert raw Newton/MuJoCo episodes (.npz from runner.py) to a LeRobot v3 dataset, optionally push it.

Run in the LeRobot venv (.venv: mujoco 3.3.7, lerobot 0.6.1), never in .venv-newton:

    python -m sim.newton.fr3.newton.convert --raw datasets/newton_raw/v1 --out datasets/cathlab_fr3_newton_v1 \
        [--limit 96] [--push aravindsuresh16/cathlab_fr3_newton_v1 --tag v1]

Schema = v1 / Siemens fr3_duo_16d (handoff A3): action and observation.state float32[16], three 224x224
videos, `success` per frame, `task` = instruction, 30 fps, robot_type fr3_duo; std = 1 for constant
columns. Episodes are written in the raw log order (= generation_order, object-balanced).
Also writes meta/episodes_extra.jsonl (full config of every attempt, with episode_index for the saved
ones) and manifest.yaml.
"""
import argparse
import getpass
import json
from pathlib import Path

import numpy as np
import yaml

from sim.newton.fr3.contracts import DatasetManifest
from sim.newton.fr3.robot_io import CONTROL_HZ, NAMES
from sim.newton.fr3.scene import CAMERAS, IMAGE_SIZE

PRESET = 12
# features() and floor_constant_stats() vendored from cathlab_sim/generate.py (which needs the v1 backend).
MIN_STD = 1e-3  # below this a column is constant (e.g. the idle right arm); its std is set to 1
MIN_QUANTILE_RANGE = 1e-2  # rad; q99 - q01 below this = idle column (left arm jitter is <= 0.0016)


def floor_constant_stats(root: Path) -> list[str]:
    """Widen the normalization stats of (near-)constant columns in meta/stats.json; return their names.

    Policies normalize with (x - mean) / (std + 1e-8) or, for quantile policies such as pi0.5,
    2 (x - q01) / (q99 - q01) - 1. For a joint that never moves (the idle arm), std and q99 - q01
    are sim jitter or rounding noise, so normalizing amplifies that noise to values of +-30..50.
    With std = 1, and q01/q99 = q50 -+ 1, those columns normalize to ~0.
    """
    path = root / "meta/stats.json"
    stats = json.loads(path.read_text())
    fixed = []
    for key in ("action", "observation.state"):
        s = stats[key]
        for j in range(len(s["std"])):
            hit = False
            if s["std"][j] < MIN_STD:
                s["std"][j] = 1.0
                hit = True
            if "q01" in s and s["q99"][j] - s["q01"][j] < MIN_QUANTILE_RANGE:
                s["q01"][j], s["q99"][j] = s["q50"][j] - 1.0, s["q50"][j] + 1.0
                hit = True
            if hit:
                fixed.append(f"{key}[{NAMES[j]}]")
    path.write_text(json.dumps(stats, indent=4))
    return fixed


def features() -> dict:
    vec = {"dtype": "float32", "shape": (16,), "names": NAMES}
    feats = {"observation.state": dict(vec), "action": dict(vec)}
    for cam in CAMERAS:
        feats[f"observation.images.{cam}"] = {"dtype": "video", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
                                              "names": ["height", "width", "channel"]}
    feats["success"] = {"dtype": "bool", "shape": (1,), "names": None}
    return feats



def raw_dirs(raw: Path) -> list[Path]:
    """`raw` itself (runner layout) or every <raw>/raw/shard_*/ (engines/newton generate layout)."""
    raw = Path(raw)
    if (raw / "meta/episodes_extra.jsonl").exists():
        return [raw]
    shards = sorted(p for p in (raw / "raw").glob("shard_*") if (p / "meta/episodes_extra.jsonl").exists())
    if not shards:
        raise FileNotFoundError(f"no meta/episodes_extra.jsonl under {raw} or {raw}/raw/shard_*/")
    return shards


def convert(raw: Path, out: Path, repo_id: str, limit: int | None = None, version: str = "v1") -> DatasetManifest:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if out.exists():
        raise FileExistsError(f"{out} exists; datasets are versioned, never overwritten")
    raws = raw_dirs(raw)
    log = [{**json.loads(line), "_raw": str(r)} for r in raws
           for line in (r / "meta/episodes_extra.jsonl").read_text().splitlines()]
    saved = [e for e in log if e["episode"]]
    if limit is not None:
        saved = saved[:limit]
        last = log.index(saved[-1])
        log = log[:last + 1]  # attempts up to the last kept episode
    from lerobot.configs.video import RGBEncoderConfig
    # Streaming encoding skips LeRobot's per-frame PNG step (~2 min per episode otherwise). Codec stays AV1
    # (libsvtav1, as v1 and the Siemens data); preset 12 = fastest SVT-AV1 speed level.
    ds = LeRobotDataset.create(repo_id=repo_id, fps=CONTROL_HZ, features=features(), root=out,
                               robot_type="fr3_duo", use_videos=True, streaming_encoding=True,
                               rgb_encoder=RGBEncoderConfig(vcodec="libsvtav1", preset=PRESET), encoder_threads=4)
    index = {}
    try:
        for i, e in enumerate(saved):
            # load every array once: indexing an NpzFile re-reads and decompresses the array on each access
            with np.load(Path(e["_raw"]) / "episodes" / f"{e['episode']}.npz") as f:
                z = {k: f[k] for k in f.files}
            task = str(z["task"])
            for t in range(len(z["action"])):
                frame = {"observation.state": z["state"][t], "action": z["action"][t],
                         "success": np.array([bool(z["success"][t])]), "task": task}
                for c in CAMERAS:
                    frame[f"observation.images.{c}"] = z[c][t]
                ds.add_frame(frame)
            ds.save_episode()
            index[e["episode"]] = i
            if (i + 1) % 16 == 0:
                print(f"  converted {i + 1}/{len(saved)}", flush=True)
    finally:
        ds.finalize()
    fixed = floor_constant_stats(out)
    with open(out / "meta/episodes_extra.jsonl", "w") as f:
        for e in log:
            e = {k: v for k, v in e.items() if k != "_raw"}
            f.write(json.dumps({"episode_index": index.get(e["episode"]), **e}) + "\n")
    per_obj = {o: f"{sum(e['success'] for e in log if e['object'] == o)}/{sum(e['object'] == o for e in log)}"
               for o in ("bottle", "vial", "syringe")}
    engines = sorted({e["engine"] for e in log})
    manifest = DatasetManifest(
        dataset_id=repo_id, version=version, source="scripted", task_id="hand_me_object",
        scene_id="cathlab_fr3_newton", n_episodes=len(saved), created_by=getpass.getuser(),
        quality_score=round(sum(e["success"] for e in log) / max(len(log), 1), 4),
        notes=(f"FR3 Duo, right arm, scripted IK expert (ground-truth poses) planned on CPU MuJoCo; physics + "
               f"rendering engine {engines}. Team variation grid (configs/variation/cathlab_fr3_newton.yaml), "
               f"configs in generation_order. attempts={len(log)}; expert success per object {per_obj}. "
               f"No-op frames dropped (handoff A6); constant-column std set to 1: {len(fixed)} columns."))
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest.model_dump(), sort_keys=False))
    print(f"wrote {len(saved)} episodes -> {out}")
    return manifest


def push(out: Path, repo_id: str, tag: str):
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id, root=out)
    # uploads the whole root (incl. manifest.yaml, meta/episodes_extra.jsonl) and tags the codebase version (v3.0)
    ds.push_to_hub(tags=["fr3_duo", "newton", "mujoco-warp", "synthetic"], private=False, tag_version=True)
    api = HfApi()
    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset")
    files = api.list_repo_files(repo_id, repo_type="dataset", revision=tag or "v3.0")
    assert "manifest.yaml" in files and "meta/episodes_extra.jsonl" in files, "extra files missing on the Hub"
    print(f"pushed https://huggingface.co/datasets/{repo_id} (tags: v3.0{', ' + tag if tag else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo-id", default="aravindsuresh16/cathlab_fr3_newton_v1")
    ap.add_argument("--limit", type=int, default=None, help="first N saved episodes (generation order)")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    convert(args.raw, args.out, args.repo_id, args.limit, args.version)
    if args.push:
        push(args.out, args.repo_id, args.tag)


if __name__ == "__main__":
    main()
