"""A tiny engine for pipeline tests: its stages write small files into --out."""
from pathlib import Path
from engines.cli_base import Flag, Stage, run_cli
from engines.registry import load_manifest

MANIFEST = load_manifest(Path(__file__).parent / "fake" / "engine.yaml")


def preview(args, ctx):
    print(f"preview n={args.n} objects={','.join(ctx.objects)}", flush=True)
    (ctx.out / "grid.txt").write_text(f"n={args.n}")


def generate(args, ctx):
    (ctx.out / "raw").mkdir()
    (ctx.out / "raw" / "ok").write_text("1")


def convert(args, ctx):
    (ctx.out / "lerobot.txt").write_text(args.dataset)


STAGES = {
    "preview": Stage(preview, flags=(Flag("--n", int, 8),)),
    "check-expert": Stage(lambda a, c: None, flags=(Flag("--n", int, 5), Flag("--split", default="train"))),
    "generate": Stage(generate, flags=(Flag("--n", int, 30), Flag("--start", int, 0), Flag("--split", default="train"))),
    "convert": Stage(convert, flags=(Flag("--dataset", required=True),)),
}

if __name__ == "__main__":
    raise SystemExit(run_cli("fake", STAGES, manifest=MANIFEST))
