"""Uniform engine CLI. Each engine's cli.py declares its stages; this module does argparse, run.yaml and errors.

    python -m engines.<id>.cli <stage> --out DIR [--run var/runs/<id>/run.yaml] [stage flags] [--dry-run]

Contract (docs/EXTENDING.md): exit 0 on success; non-zero with a one-line reason on stderr; progress on stdout;
outputs only under --out. --dry-run prints each command a stage would run (one JSON argv per line), runs nothing.
"""
import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from engines.registry import ROOT, EngineManifest, get_engine


@dataclass(frozen=True)
class Flag:
    name: str                              # "--n"
    type: Callable[[str], Any] = str
    default: Any = None
    help: str = ""
    action: str | None = None              # "append" | "store_true"
    required: bool = False

    def add_to(self, parser: argparse.ArgumentParser) -> None:
        if self.action == "store_true":
            parser.add_argument(self.name, action="store_true", help=self.help)
        elif self.action == "append":
            parser.add_argument(self.name, action="append", default=None, help=self.help)
        else:
            parser.add_argument(self.name, type=self.type, default=self.default, help=self.help,
                                required=self.required)


@dataclass
class StageContext:
    manifest: EngineManifest
    out: Path
    run: dict | None = None
    dry_run: bool = False
    root: Path = ROOT

    @property
    def selection(self) -> dict:
        return (self.run or {}).get("selection") or {}

    @property
    def objects(self) -> list[str]:
        """Objects of the run's selection; empty without a run (the stage default applies)."""
        objs = self.selection.get("objects") or []
        return [objs] if isinstance(objs, str) else list(objs)

    def path(self, rel: str) -> str:
        p = Path(rel)
        return str(p if p.is_absolute() else self.root / p)

    def run_script(self, argv: list, env: dict[str, str] | None = None) -> int:
        """Run [sys.executable, *argv] from the repo root; with --dry-run only print the argv as JSON."""
        cmd = [sys.executable, *[str(a) for a in argv]]
        if self.dry_run:
            print(json.dumps(cmd), flush=True)
            return 0
        print("$ " + " ".join(cmd), flush=True)
        return subprocess.run(cmd, cwd=self.root, env={**os.environ, **(env or {})}).returncode


@dataclass(frozen=True)
class Stage:
    fn: Callable[[argparse.Namespace, StageContext], "int | None"]
    help: str = ""
    flags: tuple = ()


def build_parser(engine_id: str, stages: dict[str, Stage]) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"python -m engines.{engine_id}.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, stage in stages.items():
        p = sub.add_parser(name, help=stage.help, description=stage.help)
        p.add_argument("--out", required=True, help="output folder (created)")
        p.add_argument("--run", help="Control Station run snapshot (run.yaml); selection.objects restricts objects")
        p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
        for flag in stage.flags:
            flag.add_to(p)
    return ap


def run_cli(engine_id: str, stages: dict[str, Stage], argv: list[str] | None = None,
            manifest: EngineManifest | None = None) -> int:
    args = build_parser(engine_id, stages).parse_args(argv)
    for flag in stages[args.stage].flags:      # append flags default to an empty list, never a shared one
        dest = flag.name.lstrip("-").replace("-", "_")
        if flag.action == "append" and getattr(args, dest) is None:
            setattr(args, dest, [])
    try:
        manifest = manifest or get_engine(engine_id)
        out = Path(args.out)
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
        run = yaml.safe_load(Path(args.run).read_text()) if args.run else None
        rc = stages[args.stage].fn(args, StageContext(manifest, out, run, args.dry_run))
        rc = 0 if rc is None else int(rc)
    except Exception as e:  # noqa: BLE001 - engine contract: one-line reason on stderr, non-zero exit
        traceback.print_exc(file=sys.stdout)
        print(f"{engine_id} {args.stage} failed: {e}", file=sys.stderr)
        return 1
    if rc != 0:
        print(f"{engine_id} {args.stage} failed: exit code {rc}", file=sys.stderr)
    return rc
