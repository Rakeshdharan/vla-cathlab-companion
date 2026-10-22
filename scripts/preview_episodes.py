"""Write a side-by-side (agentview | eye_in_hand) preview of raw .npz episodes as GIF or MP4.

    python scripts/preview_episodes.py data/pick_robosuite_base --episodes 0 1 2 --out outputs/preview.gif
"""
import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataengine.npz_sink import merge_raw_shards  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="folder passed as --out to robosuite_generate.py")
    ap.add_argument("--episodes", type=int, nargs="*", default=[0], help="merged episode indices")
    ap.add_argument("--out", default="outputs/preview.gif", help=".gif or .mp4 (mp4 needs imageio-ffmpeg)")
    ap.add_argument("--every", type=int, default=2, help="keep every k-th frame")
    args = ap.parse_args()

    shards = sorted(p for p in (Path(args.dataset) / "raw").iterdir() if (p / "manifest.yaml").exists())
    wanted = {m["episode_index"]: (m, npz) for m, npz in merge_raw_shards(shards) if m["episode_index"] in args.episodes}
    frames = []
    for idx in args.episodes:
        meta, npz = wanted[idx]
        with np.load(npz) as d:
            frames += list(np.concatenate([d["agentview"], d["eye_in_hand"]], axis=2)[::args.every])
        print(f"episode {idx}: {meta['instruction']!r}, {meta['steps']} steps, config {meta['config']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fps = 20 / args.every
    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, frames, duration=1 / fps, loop=0)
    else:
        imageio.mimsave(args.out, frames, fps=fps)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
