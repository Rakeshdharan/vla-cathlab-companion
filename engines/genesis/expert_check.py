"""Measure scripted-expert success in Genesis (no rendering, no recording); robosuite_expert_check.py's format.

    PYTHONPATH=. ~/venvs/genesis/bin/python -m engines.genesis.expert_check --n 20 --json out/expert_check.json
"""
import argparse
import dataclasses
import json
import time
from collections import Counter
from pathlib import Path

from dataengine.expert import ExpertParams, parse_param_overrides, run_expert_episode
from dataengine.variation import TIERS, VariationSpec, eval_configs, train_configs
from engines.registry import ROOT


def parse_params(pairs: list[str], base: ExpertParams) -> ExpertParams:
    try:
        return parse_param_overrides(pairs, base)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def pick_configs(spec: VariationSpec, split: str, n: int, objects: list[str]) -> list:
    if split == "train":
        cfgs = train_configs(spec)
        out = []
        for obj in objects:      # n configs per object, spread evenly over that object's grid cells
            mine = [c for c in cfgs if c.object == obj]
            out += [mine[round(i * len(mine) / n)] for i in range(n)]
        return out
    if split not in TIERS:
        raise SystemExit(f"--split must be 'train' or one of {TIERS}")
    return [c for c in eval_configs(spec, split, list(range(10_000, 10_000 + n))) if c.object in objects]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="episodes per object")
    ap.add_argument("--split", default="train", help="'train' or an eval tier")
    ap.add_argument("--objects", default=None, help="comma-separated subset (default: all)")
    ap.add_argument("--param", action="append", default=[], help="ExpertParams override, e.g. pos_tol=0.01")
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    ap.add_argument("--json", help="also write the summary here")
    args = ap.parse_args(argv)

    from sim.genesis.expert_sim import GenesisExpertSim
    from sim.genesis.params import GENESIS_EXPERT_PARAMS

    spec = VariationSpec.load(args.spec)
    params = parse_params(args.param, GENESIS_EXPERT_PARAMS)
    objects = args.objects.split(",") if args.objects else list(spec.objects)
    sim = GenesisExpertSim(spec, render=False)
    per_object: dict[str, list[int]] = {}
    steps: dict[str, list[int]] = {}
    stuck_at: dict[str, Counter] = {}
    t0 = time.time()
    for cfg in pick_configs(spec, args.split, args.n, objects):
        r = run_expert_episode(sim, spec, cfg, params)
        counts = per_object.setdefault(cfg.object, [0, 0])
        counts[0] += 1
        counts[1] += r.success
        steps.setdefault(cfg.object, []).append(r.steps)
        if not r.success:
            failed = f"stuck_after_{r.trace[-1]}" if r.trace else "stuck_at_start"
            if r.trace and r.trace[-1] == "retreat":
                failed = "not_placed"
            stuck_at.setdefault(cfg.object, Counter())[failed] += 1
            print(f"  FAIL {cfg.object} layout={cfg.layout_id} seed={cfg.seed} steps={r.steps} trace={r.trace} "
                  f"flags={r.final_info.stage_flags if r.final_info else None}", flush=True)
    elapsed = time.time() - t0
    n_eps = sum(c[0] for c in per_object.values())
    summary = {"split": args.split, "params": dataclasses.asdict(params),
               "per_object": {o: {"success": s, "attempted": a, "rate": s / a, "mean_steps": sum(steps[o]) / a,
                                  "max_steps": max(steps[o])} for o, (a, s) in per_object.items()},
               "failures": {o: dict(c) for o, c in stuck_at.items()}, "sec_per_episode": elapsed / max(n_eps, 1)}
    for obj, v in summary["per_object"].items():
        print(f"{obj:8s} {v['success']:3d}/{v['attempted']:<3d} {v['rate']:.0%}  steps mean {v['mean_steps']:.0f} "
              f"max {v['max_steps']}  failures: {summary['failures'].get(obj, {})}")
    print(f"{elapsed / max(n_eps, 1):.2f} s/episode (no rendering)")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
