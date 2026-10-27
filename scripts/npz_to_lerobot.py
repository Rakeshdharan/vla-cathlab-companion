"""Convert raw .npz shards (scripts/robosuite_generate.py) into one LeRobot v3 dataset. Run in the `lerobot` env:

    python scripts/npz_to_lerobot.py --dataset data/pick_robosuite_base
    # -> data/pick_robosuite_base/lerobot/ (LeRobot dataset + meta/episodes_extra.jsonl + manifest.yaml)

Shards are merged in name order and episodes re-indexed from 0. Each episode_extra line keeps `raw_shard`.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import DatasetManifest, load_yaml, save_yaml  # noqa: E402
from dataengine.npz_sink import iter_npz_frames, merge_raw_shards  # noqa: E402


def merged_manifest(shards: list[Path], n_episodes: int) -> DatasetManifest:
    parts = [load_yaml(s / "manifest.yaml", DatasetManifest) for s in shards]
    notes = "; ".join(f"{s.name}: {p.notes}" for s, p in zip(shards, parts, strict=True))
    return parts[0].model_copy(update={"n_episodes": n_episodes, "notes": notes})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="folder passed as --out to robosuite_generate.py")
    ap.add_argument("--root", help="LeRobot output folder (default <dataset>/lerobot, must not exist)")
    ap.add_argument("--repo-id", help="default: dataset_id from the shard manifests")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    from dataengine.lerobot_sink import LeRobotSink

    shards = sorted(p for p in (Path(args.dataset) / "raw").iterdir() if (p / "manifest.yaml").exists())
    if not shards:
        raise SystemExit(f"no shards with manifest.yaml under {args.dataset}/raw")
    root = Path(args.root or Path(args.dataset) / "lerobot")
    first = load_yaml(shards[0] / "manifest.yaml", DatasetManifest)
    sink = LeRobotSink(args.repo_id or first.dataset_id, root, fps=args.fps)
    metas = []
    for meta, npz in merge_raw_shards(shards):
        for frame in iter_npz_frames(npz):
            sink.add_frame(frame)
        sink.save_episode()
        metas.append(meta)
        print(f"episode {meta['episode_index']}: {meta['config']['object']} ({meta['steps']} frames)", flush=True)
    sink.finalize()
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta" / "episodes_extra.jsonl").write_text("".join(json.dumps(m) + "\n" for m in metas))
    save_yaml(merged_manifest(shards, len(metas)), root / "manifest.yaml")
    print(f"wrote {len(metas)} episodes from {len(shards)} shard(s) to {root}")


if __name__ == "__main__":
    main()
