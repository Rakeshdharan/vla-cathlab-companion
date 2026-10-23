"""Exit 1 unless every object in a robosuite_expert_check.py --json summary reaches the minimum success rate."""
import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    ap.add_argument("--min", type=float, default=0.8)
    args = ap.parse_args(argv)
    per_object = json.loads(Path(args.summary).read_text())["per_object"]
    low = {o: v["rate"] for o, v in per_object.items() if v["rate"] < args.min}
    for o, v in per_object.items():
        print(f"{o:12s} {v['rate']:.0%} ({v['success']}/{v['attempted']})")
    if low:
        print(f"FAIL: below {args.min:.0%}: {low}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
