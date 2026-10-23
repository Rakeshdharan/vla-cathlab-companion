"""Generate scripted robosuite episodes as raw .npz shards (convert with scripts/npz_to_lerobot.py).

    MUJOCO_GL=egl python scripts/robosuite_generate.py --split train --start 0 --n 20 --out data/pick_robosuite_base
    # parallel: one process per shard, same --start/--n, different --shard
    MUJOCO_GL=egl python scripts/robosuite_generate.py --n 384 --out data/pick_robosuite --shard 0 --num-shards 4

Configs follow dataengine.variation.generation_order (objects interleaved, seeded shuffle per object), so any
--start/--n window and every shard is object-balanced and varied, the same window is generated in every engine, and
--n 384 covers the whole train grid. Each shard writes <out>/raw/shard_<i>-of-<k>/ with
episodes/*.npz, meta/episodes_extra.jsonl and manifest.yaml. Only successful expert episodes are kept.
"""
import argparse
import getpass
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import TaskSpec, load_yaml  # noqa: E402
from dataengine.generate import generate_scripted  # noqa: E402
from dataengine.npz_sink import NpzSink  # noqa: E402
from dataengine.variation import VariationSpec, select_configs  # noqa: E402




def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="'train', an eval tier, or cells:<next_run.json>")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20, help="configs in the window (all shards together)")
    ap.add_argument("--out", required=True, help="dataset folder; raw shards go to <out>/raw/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dataset-id", default="cathlab/pick_robosuite")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--created-by", default=getpass.getuser())
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    ap.add_argument("--task", default=str(ROOT / "configs/tasks/cathlab_pick.yaml"))
    args = ap.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be in [0, --num-shards)")

    from sim.robosuite.expert_sim import RobosuiteExpertSim
    from sim.robosuite.params import ROBOSUITE_EXPERT_PARAMS

    spec = VariationSpec.load(args.spec)
    task = load_yaml(args.task, TaskSpec)
    cfgs = select_configs(spec, args.split, args.start, args.n, args.shard, args.num_shards)
    shard_dir = Path(args.out) / "raw" / f"shard_{args.shard:03d}-of-{args.num_shards:03d}"
    sink = NpzSink(shard_dir)
    sim = RobosuiteExpertSim(spec)
    print(f"shard {args.shard}/{args.num_shards}: {len(cfgs)} configs -> {shard_dir}", flush=True)
    t0 = time.time()
    report, manifest = generate_scripted(sim, spec, task, cfgs, sink, shard_dir, "robosuite", args.dataset_id,
                                         args.version, args.created_by, params=ROBOSUITE_EXPERT_PARAMS)
    minutes = (time.time() - t0) / 60
    print(f"kept {report.succeeded}/{report.attempted} ({report.success_rate:.0%}); per object "
          f"{ {o: f'{s}/{a}' for o, (a, s) in report.per_object.items()} }")
    print(f"{report.attempted / max(minutes, 1e-9):.1f} episodes/min attempted ({minutes:.1f} min)")


if __name__ == "__main__":
    main()
