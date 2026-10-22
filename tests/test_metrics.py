"""Tests for eval/metrics.py extended metrics."""
from contracts import EpisodeResult, EvalReport
from eval.metrics import (
    wilson_ci, success_by_tier, success_by_object,
    time_to_success, latency_summary, full_summary,
)


def _make_report(episodes):
    return EvalReport(
        model_version="test@v0",
        dataset_versions=[],
        backend="dummy",
        episodes=episodes,
    )


def _ep(i, ok, obj="guidewire", variant="in_distribution", stage=None, steps=10, lat50=20.0, lat95=40.0):
    return EpisodeResult(
        episode=i, seed=i, object=obj, instruction=f"hand me the {obj}",
        success=ok, failed_stage=stage, steps=steps,
        latency_ms_p50=lat50, latency_ms_p95=lat95, scene_variant=variant,
    )


def test_wilson_ci_half():
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_wilson_ci_perfect():
    lo, hi = wilson_ci(10, 10)
    assert lo > 0.6 and hi == 1.0


def test_wilson_ci_zero_n():
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 0.0


def test_success_by_tier():
    eps = [
        _ep(0, True, variant="in_distribution"),
        _ep(1, False, variant="in_distribution"),
        _ep(2, True, variant="heavy_clutter"),
    ]
    result = success_by_tier(_make_report(eps))
    assert result["in_distribution"]["success_rate"] == 0.5
    assert result["heavy_clutter"]["success_rate"] == 1.0
    assert result["in_distribution"]["n"] == 2


def test_success_by_object():
    eps = [
        _ep(0, True, obj="guidewire"),
        _ep(1, False, obj="catheter"),
        _ep(2, True, obj="guidewire"),
    ]
    result = success_by_object(_make_report(eps))
    assert result["guidewire"]["success_rate"] == 1.0
    assert result["catheter"]["success_rate"] == 0.0


def test_time_to_success():
    eps = [_ep(i, True, steps=10 + i) for i in range(5)]
    t = time_to_success(_make_report(eps))
    assert t["n_success"] == 5
    assert t["mean_steps"] == 12.0


def test_time_to_success_no_successes():
    eps = [_ep(0, False, stage="grasp")]
    t = time_to_success(_make_report(eps))
    assert t["n_success"] == 0
    assert t["mean_steps"] is None


def test_latency_summary():
    eps = [_ep(i, True, lat50=float(20 + i), lat95=float(40 + i)) for i in range(4)]
    lat = latency_summary(_make_report(eps))
    assert lat["p50_ms"] > 0
    assert lat["p95_ms"] >= lat["p50_ms"]


def test_full_summary_keys():
    eps = [_ep(i, i % 2 == 0) for i in range(6)]
    s = full_summary(_make_report(eps))
    assert all(k in s for k in [
        "success_rate", "ci_low", "ci_high", "stage_failures",
        "by_tier", "by_object", "time_to_success", "latency",
        "language_grounding_accuracy", "collision_rate",
    ])
