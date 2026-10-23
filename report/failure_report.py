"""Failure report from engine eval reports, and the targeted next run.

    python -m report.failure_report --eval robosuite=<eval_report.json> [--eval maniskill=...] \\
        --spec configs/variation/cathlab_v1.yaml --model cathlab-pick@v1 --out var/reports/<id> [--top 10] [--k 8]

A cell is (object, layout_id, obj_color_id, table_color_id), rebuilt from each episode's tier + seed with
dataengine.variation.eval_configs, so it is the same cell in every engine. next_run.json lists only cells whose
values are all in the spec's train splits: training on held-out values would leak the eval.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from contracts import STAGES, EvalReport
from dataengine.variation import TIERS, VariationSpec, eval_configs
from eval.metrics import wilson_ci

GENERATE_STAGES = ("reach", "grasp", "place")
EXPERT_STAGES = ("lift", "transport")
COLOUR_TIERS = ("held_out_obj_color", "held_out_table_color")


def cell_of(spec: VariationSpec, ep) -> dict | None:
    if ep.scene_variant not in TIERS:
        return None
    cfg = next((c for c in eval_configs(spec, ep.scene_variant, [ep.seed]) if c.object == ep.object), None)
    if cfg is None:
        return None
    return {"tier": ep.scene_variant, "object": ep.object, "layout_id": cfg.layout_id,
            "obj_color_id": cfg.obj_color_id, "table_color_id": cfg.table_color_id}


def _rate(successes: int, n: int) -> dict:
    lo, hi = wilson_ci(successes, n) if n else (0.0, 0.0)
    return {"successes": successes, "n": n, "rate": round(successes / n, 4) if n else 0.0,
            "ci": [round(lo, 4), round(hi, 4)]}


def _in_train(spec: VariationSpec, cell: dict) -> bool:
    return (cell["layout_id"] in spec.layout_split.train and cell["obj_color_id"] in spec.obj_color_split.train
            and cell["table_color_id"] in spec.table_color_split.train)


def _recommend(stage: str, tiers: list[str], k: int, trainable: bool) -> str:
    if not trainable:
        if set(tiers) <= set(COLOUR_TIERS):
            return "widen the colour palette (object/table colours) in the next spec version"
        return "widen the layout/colour ranges in the next spec version (held-out values are never trained on)"
    if stage in EXPERT_STAGES:
        return "check expert params / friction range, then generate targeted episodes"
    return f"generate {k} targeted episodes"


def build_report(evals: dict[str, EvalReport], spec: VariationSpec, spec_path: str, model: str, top: int = 10,
                 k: int = 8, primary: str = "robosuite") -> dict:
    overall, by_cell, tier_rates, stages, wrong = {}, [], defaultdict(dict), {}, {}
    cells: dict[tuple, dict] = {}
    for engine, rep in evals.items():
        eps = rep.episodes
        overall[engine] = _rate(sum(e.success for e in eps), len(eps))
        stages[engine] = {s: sum(1 for e in eps if not e.success and e.failed_stage == s) for s in STAGES}
        wrong[engine] = sum(1 for e in eps if e.judge_verdict and "wrong" in e.judge_verdict.lower())
        groups = defaultdict(list)
        for e in eps:
            groups[(e.scene_variant, e.object)].append(e)
        for (tier, obj), g in sorted(groups.items()):
            by_cell.append({"engine": engine, "tier": tier, "object": obj, **_rate(sum(e.success for e in g), len(g))})
        for tier in TIERS:
            g = [e for e in eps if e.scene_variant == tier]
            if g:
                tier_rates[tier][engine] = sum(e.success for e in g) / len(g)
        for e in eps:
            if e.success:
                continue
            cell = cell_of(spec, e)
            if cell is None:
                continue
            key = (cell["object"], cell["layout_id"], cell["obj_color_id"], cell["table_color_id"])
            c = cells.setdefault(key, {"object": key[0], "layout_id": key[1], "obj_color_id": key[2],
                                       "table_color_id": key[3], "failures": 0, "engines": Counter(),
                                       "stages": Counter(), "tiers": [], "video": None})
            c["failures"] += 1
            c["engines"][engine] += 1
            c["stages"][e.failed_stage or "reach"] += 1
            if cell["tier"] not in c["tiers"]:
                c["tiers"].append(cell["tier"])
            c["video"] = c["video"] or e.video_path
    failing = sorted(cells.values(), key=lambda c: (-c["failures"], c["object"], c["layout_id"],
                                                    c["obj_color_id"], c["table_color_id"]))[:top]
    for c in failing:
        stage = c["stages"].most_common(1)[0][0]
        c["trainable"] = _in_train(spec, c)
        c["main_stage"] = stage
        c["recommendation"] = _recommend(stage, c["tiers"], k, c["trainable"])
        c["engines"], c["stages"] = dict(c["engines"]), dict(c["stages"])
    gap = []
    for tier in TIERS:
        rates = tier_rates.get(tier, {})
        if primary in rates and len(rates) > 1:
            gap.append({"tier": tier, "primary": round(rates[primary], 4),
                        "gaps": {e: round(rates[primary] - r, 4) for e, r in rates.items() if e != primary}})
    return {"model": model, "spec": spec_path, "primary": primary, "k": k, "created": time.time(),
            "overall": overall, "by_tier_object": by_cell, "stage_failures": stages, "wrong_object": wrong,
            "sim2sim_gap": gap, "failing_cells": failing}


def next_run(report: dict) -> dict:
    cells = [{"object": c["object"], "layout_id": c["layout_id"], "obj_color_id": c["obj_color_id"],
              "table_color_id": c["table_color_id"], "k": report["k"]}
             for c in report["failing_cells"] if c["trainable"]]
    return {"spec": report["spec"], "model": report["model"], "created": report["created"], "k": report["k"],
            "cells": cells}


def render_markdown(r: dict) -> str:
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    lines = [f"# Failure report: {r['model']}", "", f"Spec `{r['spec']}`, primary engine `{r['primary']}`.", "",
             "## Overall", "", "| Engine | Success | 95 % CI | n |", "|---|---|---|---|"]
    for e, o in r["overall"].items():
        lines.append(f"| {e} | {pct(o['rate'])} | {pct(o['ci'][0])}–{pct(o['ci'][1])} | {o['n']} |")
    lines += ["", "## Failure stage", "", "| Engine | " + " | ".join(STAGES) + " | wrong object |",
              "|---" * (len(STAGES) + 2) + "|"]
    for e, h in r["stage_failures"].items():
        lines.append(f"| {e} | " + " | ".join(str(h[s]) for s in STAGES) + f" | {r['wrong_object'][e]} |")
    lines += ["", "## Sim2sim gap", "", f"Primary ({r['primary']}) minus each other engine, per tier.", ""]
    if r["sim2sim_gap"]:
        lines += ["| Tier | " + r["primary"] + " | gaps |", "|---|---|---|"]
        for g in r["sim2sim_gap"]:
            gaps = ", ".join(f"{e}: {v:+.0%}" for e, v in g["gaps"].items())
            lines.append(f"| {g['tier']} | {pct(g['primary'])} | {gaps} |")
    else:
        lines.append("Only one engine evaluated.")
    lines += ["", "## Top failing cells", "",
              "| Object | Layout | Obj colour | Table colour | Failures | Main stage | Tiers | Recommendation | Video |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in r["failing_cells"]:
        lines.append(f"| {c['object']} | {c['layout_id']} | {c['obj_color_id']} | {c['table_color_id']} | "
                     f"{c['failures']} | {c['main_stage']} | {', '.join(c['tiers'])} | {c['recommendation']} | "
                     f"{c['video'] or ''} |")
    nr = next_run(r)
    lines += ["", "## Next run", "",
              f"`next_run.json`: {len(nr['cells'])} train-split cells × {r['k']} episodes = "
              f"{len(nr['cells']) * r['k']} targeted episodes. Held-out cells are never trained on.", ""]
    return "\n".join(lines)


def write_outputs(report: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report.json": out / "report.json", "report.md": out / "report.md", "next_run.json": out / "next_run.json"}
    paths["report.json"].write_text(json.dumps(report, indent=2))
    paths["report.md"].write_text(render_markdown(report))
    paths["next_run.json"].write_text(json.dumps(next_run(report), indent=2))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m report.failure_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="engine=path/to/eval_report.json (repeatable)")
    ap.add_argument("--spec", default="configs/variation/cathlab_v1.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--primary", default="robosuite")
    a = ap.parse_args(argv)
    try:
        evals = {e: EvalReport.load(p) for e, p in (pair.split("=", 1) for pair in a.eval)}
    except (OSError, ValueError) as e:
        print(f"failure_report: {e}", file=sys.stderr)
        return 1
    report = build_report(evals, VariationSpec.load(a.spec), a.spec, a.model, a.top, a.k, a.primary)
    paths = write_outputs(report, a.out)
    print(f"wrote {', '.join(str(p) for p in paths.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
