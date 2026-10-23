"""robosuite engine CLI: the robosuite scripts behind the uniform engine contract (engines/cli_base.py).

    MUJOCO_GL=egl conda run -n robosuite python -m engines.robosuite.cli preview --out var/smoke/preview
    conda run -n lerobot python -m engines.robosuite.cli convert --dataset <generate --out> --out <dir>

Only stage functions live here; robosuite and lerobot are imported by the wrapped scripts or lazily.
"""
from engines.cli_base import Flag, Stage, run_cli

ENGINE = "robosuite"
SPEC = Flag("--spec", help="variation spec (default: the manifest's variation)")
OBJECTS = Flag("--objects", help="comma-separated objects (default: the run's objects, else all)")


def _spec(args, ctx) -> str:
    return args.spec or ctx.path(ctx.manifest.variation)


def _objects(args, ctx) -> str | None:
    return args.objects or (",".join(ctx.objects) if ctx.objects else None)


def preview(args, ctx):
    return ctx.run_script(["scripts/robosuite_variation_grid.py", "--out", ctx.out / "grid.png", "--n", args.n,
                           "--spec", _spec(args, ctx)])


def check_expert(args, ctx):
    argv = ["scripts/robosuite_expert_check.py", "--n", args.n, "--split", args.split, "--spec", _spec(args, ctx),
            "--json", ctx.out / "expert_check.json"]
    objects = _objects(args, ctx)
    if objects:
        argv += ["--objects", objects]
    for p in args.param:
        argv += ["--param", p]
    return ctx.run_script(argv)


def generate(args, ctx):
    argv = ["scripts/robosuite_generate.py", "--split", args.split, "--start", args.start, "--n", args.n,
            "--out", ctx.out, "--shard", args.shard, "--num-shards", args.num_shards,
            "--dataset-id", args.dataset_id, "--version", args.version,
            "--spec", _spec(args, ctx), "--task", ctx.path(ctx.manifest.task)]
    if args.created_by:
        argv += ["--created-by", args.created_by]
    return ctx.run_script(argv)


def convert(args, ctx):
    argv = ["scripts/npz_to_lerobot.py", "--dataset", args.dataset, "--root", ctx.out / "lerobot",
            "--fps", ctx.manifest.io_profile.fps]
    if args.repo_id:
        argv += ["--repo-id", args.repo_id]
    return ctx.run_script(argv)


STAGES = {
    "preview": Stage(preview, "grid.png of n train configs + 1 all-held-out (agentview after settle)",
                     (Flag("--n", int, 8, "train configs in the grid"), SPEC)),
    "check-expert": Stage(check_expert, "scripted expert success per object -> expert_check.json",
                          (Flag("--n", int, 20, "episodes per object"), Flag("--split", default="train"), OBJECTS,
                           Flag("--param", action="append", help="ExpertParams override k=v (repeatable)"), SPEC)),
    "generate": Stage(generate, "successful expert episodes -> raw npz shards in --out/raw/ (all objects, balanced)",
                      (Flag("--split", default="train"), Flag("--start", int, 0, "first config of the window"),
                       Flag("--n", int, 30, "configs in the window (all shards)"), Flag("--shard", int, 0),
                       Flag("--num-shards", int, 1), Flag("--dataset-id", default="cathlab/pick_robosuite"),
                       Flag("--version", default="v1"), Flag("--created-by"), SPEC)),
    "convert": Stage(convert, "raw shards -> LeRobot v3 dataset in --out/lerobot (run in the lerobot env)",
                     (Flag("--dataset", required=True, help="the generate stage's --out folder"),
                      Flag("--repo-id", help="default: dataset_id from the shard manifests"))),
}


def main(argv: list[str] | None = None) -> int:
    return run_cli(ENGINE, STAGES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
