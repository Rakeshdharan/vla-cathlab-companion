"""Render the agentview of several EpisodeConfigs after reset + settle into one PNG grid (visual variation check).

    MUJOCO_GL=egl python scripts/robosuite_variation_grid.py --out outputs/robosuite_variation_grid.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataengine.variation import VariationSpec, eval_configs, train_configs  # noqa: E402
from sim.robosuite.expert_sim import RobosuiteExpertSim  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/robosuite_variation_grid.png")
    ap.add_argument("--n", type=int, default=8, help="train configs (spread over the grid) + 1 all_held_out")
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    args = ap.parse_args()

    spec = VariationSpec.load(args.spec)
    train = train_configs(spec)
    cfgs = [train[round(i * (len(train) - 1) / max(args.n - 2, 1)) % len(train)] for i in range(args.n - 1)]
    cfgs.append(eval_configs(spec, "all_held_out", [1000])[0])
    sim = RobosuiteExpertSim(spec)
    tiles = []
    for cfg in cfgs:
        img = Image.fromarray(sim.reset(cfg)["agentview"])
        ImageDraw.Draw(img).text((4, 4), f"{cfg.tier} {cfg.object} L{cfg.layout_id} oc{cfg.obj_color_id} "
                                         f"tc{cfg.table_color_id}\n{cfg.light_temp_k:.0f}K x{cfg.light_intensity:.2f}",
                                 fill=(255, 0, 0))
        tiles.append(np.asarray(img))
    cols = 4
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), axis=1)
            for i in range(0, len(tiles), cols)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out} ({len(cfgs)} configs)")


if __name__ == "__main__":
    main()
