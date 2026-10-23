"""Newton engine CLI (FR3 Duo, fr3_duo_16d): the uniform engine contract (engines/cli_base.py).

    ~/venvs/newton/bin/python -m engines.newton.cli check-expert --out var/smoke/newton_check
    <lerobot env python> -m engines.newton.cli convert --dataset <generate --out> --out <dir>

Only stage functions live here; newton / mujoco / warp are imported lazily by engines/newton/{stages,evaluate}.py.
"""
import json
from engines.cli_base import Flag, Stage, run_cli

ENGINE = "newton"
SPEC = Flag("--spec", help="variation spec (default: the manifest's variation)")
OBJECTS = Flag("--objects", help="comma-separated objects (default: the run's objects, else all)")
WORLDS = Flag("--worlds", int, 16, "parallel Newton worlds per batch")


def _spec(args, ctx) -> str:
    return args.spec or ctx.path(ctx.manifest.variation)


def _objects(args, ctx) -> str | None:
    return args.objects or (",".join(ctx.objects) if ctx.objects else None)


def _dry(ctx, stage: str, **kwargs) -> bool:
    if ctx.dry_run:
        print(json.dumps({"stage": stage, **{k: str(v) for k, v in kwargs.items()}}), flush=True)
    return ctx.dry_run


def preview(args, ctx):
    kwargs = dict(spec_path=_spec(args, ctx), out=ctx.out, n=args.n)
    if _dry(ctx, "preview", **kwargs):
        return 0
    from engines.newton.stages import run_preview
    run_preview(**kwargs)


def check_expert(args, ctx):
    objects = _objects(args, ctx)
    kwargs = dict(spec_path=_spec(args, ctx), n=args.n, split=args.split,
                  objects=objects.split(",") if objects else None, params=args.param,
                  json_path=ctx.out / "expert_check.json", worlds=args.worlds)
    if _dry(ctx, "check-expert", **kwargs):
        return 0
    from engines.newton.stages import run_check_expert
    run_check_expert(**kwargs)


def generate(args, ctx):
    kwargs = dict(spec_path=_spec(args, ctx), task_path=ctx.path(ctx.manifest.task), out=ctx.out, split=args.split,
                  start=args.start, n=args.n, shard=args.shard, num_shards=args.num_shards,
                  dataset_id=args.dataset_id, version=args.version, created_by=args.created_by, worlds=args.worlds)
    if _dry(ctx, "generate", **kwargs):
        return 0
    from engines.newton.stages import run_generate
    run_generate(**kwargs)


def convert(args, ctx):
    argv = ["-m", "sim.newton.fr3.newton.convert", "--raw", args.dataset, "--out", ctx.out / "lerobot",
            "--repo-id", args.repo_id or "cathlab/fr3_newton"]
    return ctx.run_script(argv)


def eval_(args, ctx):
    objects = _objects(args, ctx)
    kwargs = dict(suite_path=ctx.path(args.suite), policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), model_version=args.model_version,
                  objects=objects.split(",") if objects else None,
                  tiers=args.tiers.split(",") if args.tiers else None, max_seeds=args.max_seeds, video=args.video)
    if _dry(ctx, "eval", **kwargs):
        return 0
    from engines.newton.evaluate import run_eval
    run_eval(**kwargs)


def rollout(args, ctx):
    obj = args.object or (ctx.objects[0] if ctx.objects else "vial")
    kwargs = dict(policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), obj=obj, seed=args.seed, tier=args.tier,
                  instruction=args.instruction or None, max_steps=args.max_steps)
    if _dry(ctx, "rollout", **kwargs):
        return 0
    from engines.newton.evaluate import run_rollout
    result = run_rollout(**kwargs)
    return 0 if result["video"] else 1


STAGES = {
    "preview": Stage(preview, "grid.png of n train configs + 1 all-held-out (external camera after settle)",
                     (Flag("--n", int, 8, "train configs in the grid"), SPEC)),
    "check-expert": Stage(check_expert, "scripted FR3 expert success per object in Newton -> expert_check.json",
                          (Flag("--n", int, 20, "episodes per object"), Flag("--split", default="train"), OBJECTS,
                           Flag("--param", action="append", help="not supported by the FR3 expert; must be empty"),
                           WORLDS, SPEC)),
    "generate": Stage(generate, "successful expert episodes -> raw npz shards in --out/raw/ (all objects, balanced)",
                      (Flag("--split", default="train"), Flag("--start", int, 0, "first config of the window"),
                       Flag("--n", int, 30, "configs in the window (all shards)"), Flag("--shard", int, 0),
                       Flag("--num-shards", int, 1), Flag("--dataset-id", default="cathlab/fr3_newton"),
                       Flag("--version", default="v1"), Flag("--created-by"), WORLDS, SPEC)),
    "convert": Stage(convert, "raw shards -> LeRobot v3 dataset in --out/lerobot (FR3 schema; run in a lerobot env)",
                     (Flag("--dataset", required=True, help="the generate stage's --out folder"),
                      Flag("--repo-id", help="default: cathlab/fr3_newton"))),
    "eval": Stage(eval_, "closed-loop eval over suite tiers x objects x seeds -> eval_report.json + failure videos",
                  (Flag("--suite", default="configs/eval/suite_v2.yaml"),
                   Flag("--policy", default="expert", help="expert | random | url:<policy server> | lerobot:<ckpt>"),
                   Flag("--model-version", help="label in the report (default: the policy)"),
                   Flag("--tiers", help="comma-separated tiers (default: all in the suite)"),
                   Flag("--max-seeds", int, None, help="seeds per tier (default: all)"),
                   Flag("--video", default="failures", help="none | failures | all"), OBJECTS, SPEC)),
    "rollout": Stage(rollout, "one live episode -> rollout.mp4/gif + rollout.json",
                     (Flag("--policy", default="expert"), Flag("--object", help="default: first run object, else vial"),
                      Flag("--seed", int, 5000), Flag("--tier", default="in_distribution"),
                      Flag("--instruction", help="override, e.g. from the ASSIST panel"),
                      Flag("--max-steps", int, 600, help="30 Hz frames (suite max_steps x 1.5)"), SPEC)),
}


def main(argv: list[str] | None = None) -> int:
    return run_cli(ENGINE, STAGES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
