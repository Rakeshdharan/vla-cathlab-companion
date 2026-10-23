"""Agent harness CLI.

  python -m harness list
  python -m harness run add_object --param name=guidewire_hoop --param shape=cylinder [--extra "..."] [--agent copilot]
  python -m harness show <job_id> | diff <job_id> | push <job_id> | discard <job_id>
  python -m harness serve --port 8790        # standalone agent worker (HTTP API at /api/agent)
"""
import argparse
import sys
from pathlib import Path
from harness.factory import default_harness

ROOT = Path(__file__).resolve().parents[1]


def parse_params(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--param expects key=value, got '{pair}'")
        k, v = pair.split("=", 1)
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--no-fetch", action="store_true", help="don't `git fetch` before creating the worktree")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    run = sub.add_parser("run")
    run.add_argument("task")
    run.add_argument("--param", action="append", default=[])
    run.add_argument("--extra", default="")
    run.add_argument("--agent")
    for name in ("show", "diff", "push"):
        sub.add_parser(name).add_argument("job_id")
    discard = sub.add_parser("discard")
    discard.add_argument("job_id")
    discard.add_argument("--force", action="store_true", help="force discard even if job is running")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8790)
    args = ap.parse_args(argv)

    h = default_harness(args.repo, fetch=not args.no_fetch)
    if args.cmd == "list":
        for t in h.templates.values():
            params = ", ".join(p.name for p in t.params)
            print(f"{t.id:22s} {t.title}  [{params}]")
    elif args.cmd == "run":
        job = h.create(args.task, parse_params(args.param), args.extra, args.agent)
        print(f"job {job.id} on branch {job.branch}\nlog: {h.job_dir(job.id) / 'run.log'}")
        job = h.run(job.id)
        print(f"state: {job.state}" + (f"\nerror: {job.error}" if job.error else ""))
        if job.state == "ready_for_review":
            print(f"review: python -m harness diff {job.id}\npush:   python -m harness push {job.id}")
        return 0 if job.state == "ready_for_review" else 1
    elif args.cmd == "show":
        print(h.get(args.job_id).model_dump_json(indent=2))
    elif args.cmd == "diff":
        sys.stdout.write(h.diff(args.job_id))
    elif args.cmd == "push":
        print(h.push(args.job_id).state)
    elif args.cmd == "discard":
        print(h.discard(args.job_id, force=args.force).state)
    elif args.cmd == "serve":
        import uvicorn
        from harness.server import create_agent_app
        uvicorn.run(create_agent_app(h), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
