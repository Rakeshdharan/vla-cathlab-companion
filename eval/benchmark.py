"""
Benchmark table and plots — A5 deliverable.

Compares models × data versions × simulators.

Usage:
    python -m eval.benchmark --reports reports/ --out reports/benchmark.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_reports(reports_dir: str | Path) -> list[dict]:
    """Load all EvalReport JSONs from a directory.

    Returns list of dicts with keys: model_version, dataset_versions, backend,
    success_rate, ci_low, ci_high, stage_failures, latency_p95.
    """
    from contracts import EvalReport
    from eval.metrics import full_summary

    rows = []
    for p in sorted(Path(reports_dir).glob("*.json")):
        try:
            report = EvalReport.load(p)
            s = full_summary(report)
            rows.append({
                "model_version": report.model_version,
                "dataset_versions": ", ".join(report.dataset_versions),
                "backend": report.backend,
                "success_rate": s["success_rate"],
                "ci_low": s["ci_low"],
                "ci_high": s["ci_high"],
                "stage_failures": s["stage_failures"],
                "latency_p95_ms": s["latency"]["p95_ms"],
                "n_episodes": len(report.episodes),
            })
        except Exception as e:
            print(f"WARNING: could not load {p}: {e}")
    return rows


def render_markdown_table(rows: list[dict]) -> str:
    """Render a GitHub-flavoured markdown benchmark table."""
    if not rows:
        return "_No benchmark data found._\n"

    header = "| Model | Datasets | Sim | Success ± CI | Latency p95 (ms) | N |\n"
    sep = "|---|---|---|---|---|---|\n"
    lines = [header, sep]
    for r in rows:
        lo, hi = r["ci_low"], r["ci_high"]
        lines.append(
            f"| `{r['model_version']}` | {r['dataset_versions']} | {r['backend']} "
            f"| {r['success_rate']:.1%} [{lo:.1%}–{hi:.1%}] "
            f"| {r['latency_p95_ms']:.1f} | {r['n_episodes']} |\n"
        )
    return "".join(lines)


def render_stage_failure_table(rows: list[dict]) -> str:
    """Render per-stage failure breakdown table."""
    from contracts import STAGES

    if not rows:
        return "_No data._\n"

    stages = list(STAGES)
    header = "| Model | " + " | ".join(s.capitalize() for s in stages) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(stages)) + "|\n"
    lines = [header, sep]
    for r in rows:
        sf = r.get("stage_failures", {})
        counts = " | ".join(str(sf.get(s, 0)) for s in stages)
        lines.append(f"| `{r['model_version']}` | {counts} |\n")
    return "".join(lines)


def plot_success_rates(rows: list[dict], out_path: str | Path) -> None:
    """Save a matplotlib bar chart of success rates with CI error bars."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = [f"{r['model_version']}\n({r['backend']})" for r in rows]
        rates = [r["success_rate"] for r in rows]
        errs_lo = [r["success_rate"] - r["ci_low"] for r in rows]
        errs_hi = [r["ci_high"] - r["success_rate"] for r in rows]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.5), 5))
        ax.bar(x, rates, yerr=[errs_lo, errs_hi], capsize=5, color="#4C72B0", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("CathLab Benchmark — Success Rate with 95% Wilson CI")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        fig.tight_layout()
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"Plot saved → {out_path}")
    except ImportError:
        print("WARNING: matplotlib not installed, skipping plot.")


def run_benchmark(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark table and plots")
    parser.add_argument("--reports", default="reports", help="Directory containing EvalReport JSONs")
    parser.add_argument("--out", default="reports/benchmark.md", help="Output markdown path")
    parser.add_argument("--plot", default="reports/benchmark_success.png", help="Output plot path")
    ns = parser.parse_args(args)

    rows = load_reports(ns.reports)

    md = f"# CathLab Benchmark\n\n"
    md += "## Success Rate\n\n"
    md += render_markdown_table(rows)
    md += "\n## Stage Failure Breakdown\n\n"
    md += render_stage_failure_table(rows)
    md += f"\n![Success Rate]({Path(ns.plot).name})\n"

    Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
    Path(ns.out).write_text(md)
    print(f"Benchmark table → {ns.out}")

    plot_success_rates(rows, ns.plot)


if __name__ == "__main__":
    run_benchmark()
