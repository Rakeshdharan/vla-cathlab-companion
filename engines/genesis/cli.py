"""Genesis engine CLI: the engines.genesis modules behind the uniform engine contract (engines/cli_base.py).

    ~/venvs/genesis/bin/python -m engines.genesis.cli preview --out var/genesis/preview
    conda run -n lerobot python -m engines.genesis.cli convert --dataset <generate --out> --out <dir>

Only stage functions live here; Genesis and lerobot are imported by the wrapped modules or lazily. Run one Genesis
stage at a time per GPU (the 4090 dropped off the bus twice under parallel simulator load; history on the pre-dev branch).
"""
import json
from engines.cli_base import Flag, Stage, run_cli

ENGINE = "genesis"
SPEC = Flag("--spec", help="variation spec (default: the manifest's variation)")
OBJECTS = Flag("--objects", help="comma-separated objects (default: the run's objects, else all)")
NO_AXES = Flag("--no-genesis-axes", action="store_true",
               help="disable the Genesis-only camera jitter + light direction (on by default for preview/generate)")


def _spec(args, ctx) -> str:
    return args.spec or ctx.path(ctx.manifest.variation)


def _objects(args, ctx) -> str | None:
    return args.objects or (",".join(ctx.objects) if ctx.objects else None)


def _no_axes(args) -> list[str]:
    return ["--no-genesis-axes"] if args.no_genesis_axes else []


def preview(args, ctx):
    return ctx.run_script(["-m", "engines.genesis.variation_grid", "--out", ctx.out / "grid.png", "--n", args.n,
                           "--spec", _spec(args, ctx), *_no_axes(args)])


def check_expert(args, ctx):
    argv = ["-m", "engines.genesis.expert_check", "--n", args.n, "--split", args.split, "--spec", _spec(args, ctx),
            "--json", ctx.out / "expert_check.json"]
    objects = _objects(args, ctx)
    if objects:
        argv += ["--objects", objects]
    for p in args.param:
        argv += ["--param", p]
    return ctx.run_script(argv)


def generate(args, ctx):
    argv = ["-m", "engines.genesis.generate", "--split", args.split, "--start", args.start, "--n", args.n,
            "--out", ctx.out, "--shard", args.shard, "--num-shards", args.num_shards,
            "--dataset-id", args.dataset_id, "--version", args.version,
            "--spec", _spec(args, ctx), "--task", ctx.path(ctx.manifest.task)]
    if args.created_by:
        argv += ["--created-by", args.created_by]
    return ctx.run_script(argv + _no_axes(args))


def convert(args, ctx):
    argv = ["scripts/npz_to_lerobot.py", "--dataset", args.dataset, "--root", ctx.out / "lerobot",
            "--fps", ctx.manifest.io_profile.fps]
    if args.repo_id:
        argv += ["--repo-id", args.repo_id]
    return ctx.run_script(argv)


def _dry(ctx, stage: str, **kwargs) -> bool:
    if ctx.dry_run:
        print(json.dumps({"stage": stage, **{k: str(v) for k, v in kwargs.items()}}), flush=True)
    return ctx.dry_run


def eval_(args, ctx):
    kwargs = dict(suite_path=ctx.path(args.suite), policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), model_version=args.model_version,
                  objects=_objects(args, ctx).split(",") if _objects(args, ctx) else None,
                  tiers=args.tiers.split(",") if args.tiers else None, max_seeds=args.max_seeds, video=args.video)
    if _dry(ctx, "eval", **kwargs):
        return 0
    from engines.genesis.evaluate import run_eval
    run_eval(**kwargs)


def rollout(args, ctx):
    obj = args.object or (ctx.objects[0] if ctx.objects else "vial")
    kwargs = dict(policy_arg=args.policy, out=ctx.out, spec_path=_spec(args, ctx),
                  task_path=ctx.path(ctx.manifest.task), obj=obj, seed=args.seed, tier=args.tier,
                  instruction=args.instruction or None, max_steps=args.max_steps)
    if _dry(ctx, "rollout", **kwargs):
        return 0
    from engines.genesis.evaluate import run_rollout
    result = run_rollout(**kwargs)
    return 0 if result["video"] else 1


STAGES = {
    "preview": Stage(preview, "grid.png of n train configs + 1 all-held-out (agentview after settle)",
                     (Flag("--n", int, 8, "train configs in the grid"), SPEC, NO_AXES)),
    "check-expert": Stage(check_expert, "scripted expert success per object -> expert_check.json",
                          (Flag("--n", int, 20, "episodes per object"), Flag("--split", default="train"), OBJECTS,
                           Flag("--param", action="append", help="ExpertParams override k=v (repeatable)"), SPEC)),
    "generate": Stage(generate, "successful expert episodes -> raw npz shards in --out/raw/ (all objects, balanced)",
                      (Flag("--split", default="train"), Flag("--start", int, 0, "first config of the window"),
                       Flag("--n", int, 30, "configs in the window (all shards)"), Flag("--shard", int, 0),
                       Flag("--num-shards", int, 1), Flag("--dataset-id", default="cathlab/pick_genesis"),
                       Flag("--version", default="v1"), Flag("--created-by"), SPEC, NO_AXES)),
    "convert": Stage(convert, "raw shards -> LeRobot v3 dataset in --out/lerobot (run in the lerobot env)",
                     (Flag("--dataset", required=True, help="the generate stage's --out folder"),
                      Flag("--repo-id", help="default: dataset_id from the shard manifests"))),
    "eval": Stage(eval_, "closed-loop eval over suite tiers x objects x seeds -> eval_report.json + failure videos",
                  (Flag("--suite", default="configs/eval/suite_v2.yaml"),
                   Flag("--policy", default="expert", help="expert | random | url:<policy server>"),
                   Flag("--model-version", help="label in the report (default: the policy)"),
                   Flag("--tiers", help="comma-separated tiers (default: all in the suite)"),
                   Flag("--max-seeds", int, None, help="seeds per tier (default: all)"),
                   Flag("--video", default="failures", help="none | failures | all"), OBJECTS, SPEC)),
    "rollout": Stage(rollout, "one live episode -> rollout.mp4/gif + rollout.json",
                     (Flag("--policy", default="expert"), Flag("--object", help="default: first run object, else vial"),
                      Flag("--seed", int, 5000), Flag("--tier", default="in_distribution"),
                      Flag("--instruction", help="override, e.g. from the ASSIST panel"),
                      Flag("--max-steps", int, 400), SPEC)),
}


def main(argv: list[str] | None = None) -> int:
    return run_cli(ENGINE, STAGES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
