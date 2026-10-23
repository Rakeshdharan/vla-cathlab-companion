"""Generate scripted Genesis episodes as raw .npz shards (convert with scripts/npz_to_lerobot.py).

    PYTHONPATH=. ~/venvs/genesis/bin/python -m engines.genesis.generate --n 20 --out var/genesis/pick_genesis

Same windows and shards as scripts/robosuite_generate.py: `generation_order(spec, train_configs(spec))
[start:start+n][shard::num_shards]`, so episode i is paired with robosuite's episode i. Each shard writes
<out>/raw/shard_<i>-of-<k>/ with episodes/*.npz, meta/episodes_extra.jsonl (+ engine/physics versions and git
commit per line) and manifest.yaml. Only successful expert episodes are kept.
"""
import argparse
import getpass
import json
import platform
import subprocess
import time
from pathlib import Path

from contracts import TaskSpec, load_yaml
from dataengine.variation import TIERS, EpisodeConfig, VariationSpec, eval_configs, generation_order, train_configs
from engines.registry import ROOT


def select_configs(spec: VariationSpec, split: str, start: int, n: int, shard: int, num_shards: int,
                   eval_seed0: int = 100_000) -> list[EpisodeConfig]:
    if split == "train":
        cfgs = train_configs(spec)
    elif split in TIERS:
        cfgs = eval_configs(spec, split, list(range(eval_seed0, eval_seed0 + (start + n + 2) // 3 + 1)))
    else:
        raise SystemExit(f"--split must be 'train' or one of {TIERS}")
    return generation_order(spec, cfgs)[start:start + n][shard::num_shards]


def provenance(env) -> dict:
    """Engine/physics versions and the git commit, recorded with every episode (data-quality rule E4)."""
    from importlib.metadata import version
    import torch
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "sim", "engines", "dataengine"],
                                    cwd=ROOT, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"engine": "genesis", "genesis": version("genesis-world"), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(), "sim_dt": env.dt, "substeps": env.substeps,
            "control_hz": round(1.0 / env.control_dt), "git_commit": commit, "git_dirty": dirty}


def add_provenance(meta_path: Path, prov: dict) -> None:
    """dataengine.generate writes fixed keys per episode; add `provenance` to each line of this shard."""
    lines = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
    meta_path.write_text("".join(json.dumps({**m, "provenance": prov}) + "\n" for m in lines))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="'train' or an eval tier")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=20, help="configs in the window (all shards together)")
    ap.add_argument("--out", required=True, help="dataset folder; raw shards go to <out>/raw/")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--dataset-id", default="cathlab/pick_genesis")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--created-by", default=getpass.getuser())
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    ap.add_argument("--task", default=str(ROOT / "configs/tasks/cathlab_pick.yaml"))
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must be in [0, --num-shards)")

    from dataengine.generate import generate_scripted
    from dataengine.npz_sink import NpzSink
    from sim.genesis.expert_sim import GenesisExpertSim
    from sim.genesis.params import GENESIS_EXPERT_PARAMS

    spec = VariationSpec.load(args.spec)
    task = load_yaml(args.task, TaskSpec)
    cfgs = select_configs(spec, args.split, args.start, args.n, args.shard, args.num_shards)
    shard_dir = Path(args.out) / "raw" / f"shard_{args.shard:03d}-of-{args.num_shards:03d}"
    sink = NpzSink(shard_dir)
    sim = GenesisExpertSim(spec)
    print(f"shard {args.shard}/{args.num_shards}: {len(cfgs)} configs -> {shard_dir}", flush=True)
    t0 = time.time()
    try:
        report, _ = generate_scripted(sim, spec, task, cfgs, sink, shard_dir, "genesis", args.dataset_id,
                                      args.version, args.created_by, params=GENESIS_EXPERT_PARAMS)
        add_provenance(shard_dir / "meta" / "episodes_extra.jsonl", provenance(sim.env))
    finally:
        sim.env.close()
    minutes = (time.time() - t0) / 60
    print(f"kept {report.succeeded}/{report.attempted} ({report.success_rate:.0%}); per object "
          f"{ {o: f'{s}/{a}' for o, (a, s) in report.per_object.items()} }")
    print(f"{report.attempted / max(minutes, 1e-9):.1f} episodes/min attempted ({minutes:.1f} min)")


if __name__ == "__main__":
    main()
