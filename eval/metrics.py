"""
Extended metrics for the CathLab eval suite.

All functions consume EvalReport / list[EpisodeResult] and return
plain Python dicts/lists suitable for JSON serialisation and W&B logging.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts import EvalReport, EpisodeResult


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Returns:
        (lower, upper) bounds in [0, 1].
    """
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def success_by_tier(report: "EvalReport") -> dict[str, dict]:
    """Break down success rate and Wilson CI per scene_variant (difficulty tier).

    Returns:
        {tier_name: {success_rate, ci_low, ci_high, n}}
    """
    groups: dict[str, list["EpisodeResult"]] = defaultdict(list)
    for ep in report.episodes:
        groups[ep.scene_variant].append(ep)

    result = {}
    for tier, eps in groups.items():
        n = len(eps)
        s = sum(e.success for e in eps)
        lo, hi = wilson_ci(s, n)
        result[tier] = {"success_rate": s / max(n, 1), "ci_low": lo, "ci_high": hi, "n": n}
    return result


def success_by_object(report: "EvalReport") -> dict[str, dict]:
    """Break down success rate per target object.

    Returns:
        {object_name: {success_rate, ci_low, ci_high, n}}
    """
    groups: dict[str, list["EpisodeResult"]] = defaultdict(list)
    for ep in report.episodes:
        groups[ep.object].append(ep)

    result = {}
    for obj, eps in groups.items():
        n = len(eps)
        s = sum(e.success for e in eps)
        lo, hi = wilson_ci(s, n)
        result[obj] = {"success_rate": s / max(n, 1), "ci_low": lo, "ci_high": hi, "n": n}
    return result


def time_to_success(report: "EvalReport") -> dict:
    """Stats on steps-to-success for successful episodes.

    Returns:
        {mean_steps, median_steps, p95_steps, n_success}
    """
    import numpy as np

    steps = [e.steps for e in report.episodes if e.success]
    if not steps:
        return {"mean_steps": None, "median_steps": None, "p95_steps": None, "n_success": 0}
    arr = sorted(steps)
    return {
        "mean_steps": float(np.mean(arr)),
        "median_steps": float(np.median(arr)),
        "p95_steps": float(np.percentile(arr, 95)),
        "n_success": len(arr),
    }


def latency_summary(report: "EvalReport") -> dict:
    """Aggregate latency p50/p95 across all episodes.

    Returns:
        {p50_ms, p95_ms, max_ms}
    """
    import numpy as np

    p50s = [e.latency_ms_p50 for e in report.episodes]
    p95s = [e.latency_ms_p95 for e in report.episodes]
    if not p50s:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "p50_ms": float(np.median(p50s)),
        "p95_ms": float(np.percentile(p95s, 95)),
        "max_ms": float(max(p95s)),
    }


def language_grounding_accuracy(report: "EvalReport") -> float:
    """Fraction of episodes where the correct object was acted on.

    Uses judge_verdict field: if present and contains 'wrong object' it's a grounding failure.
    Falls back to 1.0 (all correct) when no verdict is available.
    """
    verdicts = [e.judge_verdict for e in report.episodes if e.judge_verdict is not None]
    if not verdicts:
        return 1.0
    correct = sum(1 for v in verdicts if "wrong object" not in v.lower())
    return correct / len(verdicts)


def collision_rate(report: "EvalReport") -> float:
    """Fraction of episodes with at least one collision.

    Requires judge_verdict or collision field. Uses judge_verdict keyword search as fallback.
    """
    n = len(report.episodes)
    if n == 0:
        return 0.0
    col = sum(
        1 for e in report.episodes
        if (e.judge_verdict and "collision" in e.judge_verdict.lower())
    )
    return col / n


def full_summary(report: "EvalReport") -> dict:
    """Return a single dict with all metrics, suitable for W&B logging.

    Keys:
        success_rate, ci_low, ci_high, stage_failures, by_tier, by_object,
        time_to_success, latency, language_grounding_accuracy, collision_rate
    """
    lo, hi = report.wilson_ci()
    return {
        "success_rate": report.success_rate(),
        "ci_low": lo,
        "ci_high": hi,
        "stage_failures": report.stage_failure_hist(),
        "by_tier": success_by_tier(report),
        "by_object": success_by_object(report),
        "time_to_success": time_to_success(report),
        "latency": latency_summary(report),
        "language_grounding_accuracy": language_grounding_accuracy(report),
        "collision_rate": collision_rate(report),
    }
