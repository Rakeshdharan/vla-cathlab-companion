"""ManiSkill stage implementations behind engines/maniskill/cli.py: expert check, generation and the variation preview.
Run in the `maniskill` env; mani_skill is imported lazily (inside the functions)."""
import dataclasses
import getpass
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from contracts import TaskSpec, load_yaml
from dataengine.expert import ExpertParams, run_expert_episode
from dataengine.variation import TIERS, EpisodeConfig, VariationSpec, eval_configs, select_configs, train_configs


def expert_params(pairs: list[str] | None = None) -> ExpertParams:
    from sim.maniskill.params import MANISKILL_EXPERT_PARAMS
    fields = {f.name: f.type for f in dataclasses.fields(ExpertParams)}
    overrides = {}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if key not in fields:
            raise ValueError(f"unknown ExpertParams field {key!r}; choose from {sorted(fields)}")
        overrides[key] = int(value) if fields[key] in (int, "int") else float(value)
    return dataclasses.replace(MANISKILL_EXPERT_PARAMS, **overrides)


def check_configs(spec: VariationSpec, split: str, n: int, objects: list[str]) -> list[EpisodeConfig]:
    """n configs per object: spread evenly over that object's train grid cells, or eval seeds 10000+ of a tier."""
    if split == "train":
        cfgs = train_configs(spec)
        out = []
        for obj in objects:
            mine = [c for c in cfgs if c.object == obj]
            out += [mine[round(i * len(mine) / n)] for i in range(n)]
        return out
    if split not in TIERS:
        raise ValueError(f"--split must be 'train' or one of {TIERS}")
    return [c for c in eval_configs(spec, split, list(range(10_000, 10_000 + n))) if c.object in objects]




def run_check_expert(spec_path, n: int, split: str, objects: list[str] | None, params: list[str] | None,
                     json_path: Path | None) -> dict:
    from sim.maniskill.expert_sim import ManiSkillExpertSim
    spec = VariationSpec.load(spec_path)
    p = expert_params(params)
    sim = ManiSkillExpertSim(spec, render=False)
    per_object: dict[str, list[int]] = {}
    steps: dict[str, list[int]] = {}
    stuck_at: dict[str, Counter] = {}
    t0 = time.time()
    try:
        for cfg in check_configs(spec, split, n, objects or list(spec.objects)):
            r = run_expert_episode(sim, spec, cfg, p)
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
    finally:
        sim.env.close()
    elapsed = time.time() - t0
    n_eps = sum(c[0] for c in per_object.values())
    summary = {"split": split, "engine": "maniskill", "params": dataclasses.asdict(p),
               "per_object": {o: {"success": s, "attempted": a, "rate": s / a, "mean_steps": sum(steps[o]) / a,
                                  "max_steps": max(steps[o])} for o, (a, s) in per_object.items()},
               "failures": {o: dict(c) for o, c in stuck_at.items()}, "sec_per_episode": elapsed / max(n_eps, 1)}
    for obj, v in summary["per_object"].items():
        print(f"{obj:8s} {v['success']:3d}/{v['attempted']:<3d} {v['rate']:.0%}  steps mean {v['mean_steps']:.0f} "
              f"max {v['max_steps']}  failures: {summary['failures'].get(obj, {})}", flush=True)
    print(f"{elapsed / max(n_eps, 1):.2f} s/episode (no rendering)", flush=True)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(summary, indent=2))
    return summary


def versions() -> dict:
    import mani_skill
    import sapien
    import torch
    from dataengine.quality import engine_provenance
    return engine_provenance("maniskill", mani_skill=mani_skill.__version__, sapien=sapien.__version__,
                             torch=torch.__version__, physics="PhysX 5 (CPU)")


def run_generate(spec_path, task_path, out: Path, split: str, start: int, n: int, shard: int, num_shards: int,
                 dataset_id: str, version: str, created_by: str | None) -> dict:
    from dataengine.generate import generate_scripted
    from dataengine.npz_sink import NpzSink
    from sim.maniskill.expert_sim import ManiSkillExpertSim
    if not 0 <= shard < num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")
    spec, task = VariationSpec.load(spec_path), load_yaml(task_path, TaskSpec)
    cfgs = select_configs(spec, split, start, n, shard, num_shards)
    shard_dir = Path(out) / "raw" / f"shard_{shard:03d}-of-{num_shards:03d}"
    sim = ManiSkillExpertSim(spec)
    print(f"shard {shard}/{num_shards}: {len(cfgs)} configs -> {shard_dir}", flush=True)
    t0 = time.time()
    try:
        report, _ = generate_scripted(sim, spec, task, cfgs, NpzSink(shard_dir), shard_dir, "maniskill", dataset_id,
                                      version, created_by or getpass.getuser(), params=expert_params(),
                                      extra_meta=versions())
    finally:
        sim.env.close()
    minutes = (time.time() - t0) / 60
    rate = report.attempted / max(minutes, 1e-9)
    print(f"kept {report.succeeded}/{report.attempted} ({report.success_rate:.0%}); per object "
          f"{ {o: f'{s}/{a}' for o, (a, s) in report.per_object.items()} }", flush=True)
    print(f"{rate:.1f} episodes/min attempted ({minutes:.1f} min)", flush=True)
    return {"attempted": report.attempted, "succeeded": report.succeeded, "episodes_per_min": rate}


def run_preview(spec_path, out_png: Path, n: int = 8) -> Path:
    from PIL import Image, ImageDraw
    from sim.maniskill.expert_sim import ManiSkillExpertSim
    spec = VariationSpec.load(spec_path)
    train = train_configs(spec)
    cfgs = [train[round(i * (len(train) - 1) / max(n - 2, 1)) % len(train)] for i in range(n - 1)]
    cfgs.append(eval_configs(spec, "all_held_out", [1000])[0])
    sim = ManiSkillExpertSim(spec)
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
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)
    print(f"wrote {out_png} ({len(cfgs)} configs)", flush=True)
    return out_png
