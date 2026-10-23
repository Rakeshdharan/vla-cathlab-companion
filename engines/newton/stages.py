"""Newton FR3 stages: preview, check-expert, generate (run in ~/venvs/newton). newton / mujoco are imported lazily.

The FR3 expert plans on CPU MuJoCo (a process pool) and executes in Newton, `worlds` episodes per batch:
sim.newton.fr3.newton.runner, vendored from Aravind's mujoco_cathlab_vla.
"""
import getpass
import json
import os
import time
from pathlib import Path

import numpy as np
from contracts import DatasetManifest, TaskSpec, load_yaml, save_yaml

OBJECTS = ("bottle", "vial", "syringe")
EVAL_SEED0 = 5000            # non-train splits draw seeds like configs/eval/suite_v2.yaml


def _variation(spec_path):
    from sim.newton.fr3.newton.task_scene import load_variation
    return load_variation(spec_path)


def _procs() -> int:
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def select_configs(variation, split: str, n: int, objects=None) -> list:
    """First n configs per object: train = generation_order of the train grid; else a tier with seeds 5000+."""
    from sim.newton.fr3.newton.runner import configs
    from sim.newton.fr3.newton.variation import eval_configs
    objects = list(objects or OBJECTS)
    pool = configs(0, variation) if split == "train" else eval_configs(variation, split, list(range(EVAL_SEED0, EVAL_SEED0 + n)))
    return [c for o in objects for c in [c for c in pool if c.object == o][:n]]


def plan_and_execute(variation, cfgs, worlds: int, jitters=None, record: bool = False, sim=None, pool=None) -> list[dict]:
    """Plan every config on CPU, execute in Newton batches of `worlds`; one result dict per config.
    Pass `sim` (NewtonTaskSim) and `pool` (runner.make_pool) to reuse them across calls."""
    from sim.newton.fr3.newton.runner import make_pool, plan_one, run_newton_batch
    from sim.newton.fr3.newton.scene import NewtonTaskSim
    jitters = jitters if jitters is not None else [np.zeros(2)] * len(cfgs)
    jobs = [{"cfg": c.model_dump(), "jitter": list(map(float, j))} for c, j in zip(cfgs, jitters)]
    if pool is None:
        with make_pool(_procs()) as own:
            plans = own.map(plan_one, jobs)
    else:
        plans = pool.map(plan_one, jobs)
    sim = sim or NewtonTaskSim(max(1, min(worlds, len(cfgs))), variation=variation, render=record)
    res = []
    for i in range(0, len(cfgs), sim.n):
        res += run_newton_batch(sim, cfgs[i:i + sim.n], [p["actions"] for p in plans[i:i + sim.n]],
                                jitters[i:i + sim.n], record=record)
    return res


def run_check_expert(spec_path, n: int, split: str, objects, params, json_path: Path, worlds: int = 16) -> dict:
    if params:
        raise ValueError("the FR3 expert has no ExpertParams overrides; drop --param")
    variation = _variation(spec_path)
    cfgs = select_configs(variation, split, n, objects)
    t0 = time.time()
    res = plan_and_execute(variation, cfgs, worlds)
    per_object, failures = {}, []
    for obj in dict.fromkeys(c.object for c in cfgs):
        idx = [i for i, c in enumerate(cfgs) if c.object == obj]
        steps = [res[i]["n_frames"] for i in idx]
        ok = sum(res[i]["success"] for i in idx)
        per_object[obj] = {"success": ok, "attempted": len(idx), "rate": round(ok / len(idx), 3),
                           "mean_steps": round(float(np.mean(steps)), 1), "max_steps": int(max(steps))}
    for c, r in zip(cfgs, res):
        if not r["success"]:
            failures.append({"object": c.object, "layout_id": c.layout_id, "seed": c.seed, "tier": c.tier,
                             "failed_stage": r["failure_stage"]})
    report = {"split": split, "params": {}, "engine": "newton", "worlds": min(worlds, len(cfgs)),
              "seconds": round(time.time() - t0, 1), "per_object": per_object, "failures": failures}
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("split", "per_object", "seconds")}, indent=1), flush=True)
    return report


def run_generate(spec_path, task_path, out: Path, split: str, start: int, n: int, shard: int, num_shards: int,
                 dataset_id: str, version: str, created_by: str | None, worlds: int = 16) -> dict:
    from sim.newton.fr3.newton.runner import configs, drop_noops, instruction, make_pool, save_npz, versions
    from sim.newton.fr3.newton.scene import NewtonTaskSim
    if split != "train":
        raise ValueError("generate draws from the train grid only (--split train)")
    if not 0 <= shard < num_shards:
        raise ValueError("--shard must be in [0, --num-shards)")
    variation, task = _variation(spec_path), load_yaml(task_path, TaskSpec)
    cfgs = configs(0, variation)[start:start + n][shard::num_shards]
    shard_dir = Path(out) / "raw" / f"shard_{shard:03d}-of-{num_shards:03d}"
    (shard_dir / "episodes").mkdir(parents=True, exist_ok=True)
    (shard_dir / "meta").mkdir(exist_ok=True)
    print(f"shard {shard}/{num_shards}: {len(cfgs)} configs -> {shard_dir}", flush=True)
    ver = {**versions(), "engine": "newton"}
    t0, saved, rows = time.time(), 0, []
    if not cfgs:
        raise ValueError(f"empty window: --start {start} --n {n} (shard {shard}/{num_shards})")
    sim = NewtonTaskSim(max(1, min(worlds, len(cfgs))), variation=variation, render=True)
    pool = make_pool(_procs())
    try:
        for i in range(0, len(cfgs), sim.n):
            batch = cfgs[i:i + sim.n]
            for cfg, r in zip(batch, plan_and_execute(variation, batch, sim.n, record=True, sim=sim, pool=pool)):
                name = f"{cfg.object}_l{cfg.layout_id}_o{cfg.obj_color_id}_t{cfg.table_color_id}_s{cfg.seed}"
                kept = None
                if r["success"]:
                    rec = drop_noops(r["record"], True)
                    kept = len(rec["action"])
                    save_npz(shard_dir / "episodes" / f"{name}.npz", rec, instruction(cfg))
                    saved += 1
                rows.append({"episode": name if r["success"] else None, **cfg.model_dump(),
                             "pose_jitter_m": [0.0, 0.0], "instruction": instruction(cfg), "success": r["success"],
                             "failure_stage": r["failure_stage"], "n_frames_raw": r["n_frames"],
                             "n_frames_kept": kept, **ver})
            print(f"[{min(i + sim.n, len(cfgs))}/{len(cfgs)}] saved {saved}  {time.time() - t0:.0f}s", flush=True)
    finally:
        pool.close()
    (shard_dir / "meta" / "episodes_extra.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    save_yaml(DatasetManifest(dataset_id=dataset_id, version=version, source="scripted", task_id=task.task_id,
                              scene_id="cathlab_fr3_newton", n_episodes=saved,
                              created_by=created_by or getpass.getuser(),
                              quality_score=round(saved / max(len(cfgs), 1), 4),
                              notes=f"FR3 Duo right arm, fr3_duo_16d, 30 Hz; Newton {ver.get('newton')} "
                                    f"(MuJoCo-Warp); expert planned on CPU MuJoCo; {ver['vendored_from']}"),
              shard_dir / "manifest.yaml")
    minutes = (time.time() - t0) / 60
    print(f"kept {saved}/{len(cfgs)}; {len(cfgs) / max(minutes, 1e-9):.1f} episodes/min attempted", flush=True)
    return {"attempted": len(cfgs), "succeeded": saved, "episodes_per_min": len(cfgs) / max(minutes, 1e-9)}


def run_preview(spec_path, out: Path, n: int = 8) -> Path:
    from PIL import Image, ImageDraw
    from sim.newton.fr3.newton.runner import configs
    from sim.newton.fr3.newton.scene import NewtonTaskSim
    from sim.newton.fr3.newton.variation import eval_configs
    variation = _variation(spec_path)
    train = configs(0, variation)
    cfgs = [train[round(i * (len(train) - 1) / max(n - 1, 1))] for i in range(n)]
    cfgs.append(eval_configs(variation, "all_held_out", [1000])[0])
    sim = NewtonTaskSim(len(cfgs), variation=variation, render=True)
    sim.reset(cfgs)
    imgs = sim.images()["external_camera"]
    tiles = []
    for cfg, arr in zip(cfgs, imgs):
        img = Image.fromarray(arr)
        ImageDraw.Draw(img).text((4, 4), f"{cfg.tier} {cfg.object} L{cfg.layout_id} oc{cfg.obj_color_id} "
                                         f"tc{cfg.table_color_id}\n{cfg.light_temp_k:.0f}K x{cfg.light_intensity:.2f}",
                                 fill=(255, 0, 0))
        tiles.append(np.asarray(img))
    cols = 3
    rows = [np.concatenate(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])), axis=1)
            for i in range(0, len(tiles), cols)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(rows, axis=0)).save(out / "grid.png")
    (out / "preview.json").write_text(json.dumps({"engine": "newton", "camera": "external_camera",
                                                  "configs": [c.model_dump() for c in cfgs]}, indent=2))
    print(f"wrote {out / 'grid.png'} ({len(cfgs)} configs)", flush=True)
    return out / "grid.png"
