"""Render the agentview of several EpisodeConfigs after reset + settle into one PNG grid (visual variation check).

    PYTHONPATH=. ~/venvs/genesis/bin/python -m engines.genesis.variation_grid --out var/genesis/preview/grid.png
"""
import argparse
from pathlib import Path

import numpy as np

from dataengine.variation import VariationSpec, eval_configs, train_configs
from engines.registry import ROOT


def grid_configs(spec: VariationSpec, n: int) -> list:
    """n train configs spread evenly over the grid + 1 all_held_out."""
    train = train_configs(spec)
    cfgs = [train[round(i * (len(train) - 1) / max(n - 1, 1))] for i in range(n)]
    return cfgs + [eval_configs(spec, "all_held_out", [1000])[0]]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8, help="train configs (spread over the grid); 1 all_held_out is added")
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    args = ap.parse_args(argv)

    from PIL import Image, ImageDraw
    from sim.genesis.expert_sim import GenesisExpertSim

    spec = VariationSpec.load(args.spec)
    cfgs = grid_configs(spec, args.n)
    sim = GenesisExpertSim(spec)
    tiles = []
    try:
        for cfg in cfgs:
            img = Image.fromarray(sim.reset(cfg)["agentview"])
            ImageDraw.Draw(img).text((4, 4), f"{cfg.tier} {cfg.object} L{cfg.layout_id} oc{cfg.obj_color_id} "
                                             f"tc{cfg.table_color_id}\n{cfg.light_temp_k:.0f}K "
                                             f"x{cfg.light_intensity:.2f}", fill=(255, 0, 0))
            tiles.append(np.asarray(img))
    finally:
        sim.env.close()
    cols = 4
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), axis=1)
            for i in range(0, len(tiles), cols)]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(args.out)
    print(f"wrote {args.out} ({len(cfgs)} configs)")


if __name__ == "__main__":
    main()
