"""Tests for eval/failures.py failure analysis."""
from contracts import EpisodeResult, EvalReport
from eval.failures import FailureAnalyser, GapSpec


def _ep(i, ok, obj="guidewire", variant="in_distribution", stage=None):
    return EpisodeResult(
        episode=i, seed=i, object=obj, instruction=f"hand me the {obj}",
        success=ok, failed_stage=stage, steps=10,
        latency_ms_p50=20.0, latency_ms_p95=40.0, scene_variant=variant,
    )


def _make_report(episodes):
    return EvalReport(model_version="test@v0", dataset_versions=[], backend="dummy", episodes=episodes)


def test_no_failures_returns_empty_gaps():
    report = _make_report([_ep(i, True) for i in range(5)])
    analyser = FailureAnalyser(report)
    assert analyser.to_gaps() == []


def test_single_failure_cluster():
    report = _make_report([
        _ep(0, True),
        _ep(1, False, obj="catheter", stage="grasp"),
        _ep(2, False, obj="catheter", stage="grasp"),
    ])
    analyser = FailureAnalyser(report)
    gaps = analyser.to_gaps()
    assert len(gaps) == 1
    assert gaps[0].failed_stage == "grasp"
    assert "catheter" in gaps[0].objects
    assert gaps[0].n_failures == 2


def test_multiple_clusters_sorted_by_priority():
    report = _make_report([
        _ep(0, False, obj="guidewire", stage="reach"),
        _ep(1, False, obj="guidewire", stage="reach"),
        _ep(2, False, obj="guidewire", stage="reach"),
        _ep(3, False, obj="sheath", stage="grasp"),
    ])
    analyser = FailureAnalyser(report)
    gaps = analyser.to_gaps()
    assert gaps[0].n_failures >= gaps[1].n_failures


def test_suggested_generator_grasp():
    report = _make_report([_ep(0, False, stage="grasp")])
    gaps = FailureAnalyser(report).to_gaps()
    assert gaps[0].suggested_generator == "scripted"


def test_suggested_generator_reach():
    report = _make_report([_ep(0, False, stage="reach")])
    gaps = FailureAnalyser(report).to_gaps()
    assert gaps[0].suggested_generator == "domain_rand"


def test_suggested_generator_transport():
    report = _make_report([_ep(0, False, stage="transport")])
    gaps = FailureAnalyser(report).to_gaps()
    assert gaps[0].suggested_generator == "mimicgen"


def test_gapspec_to_dict():
    report = _make_report([_ep(0, False, stage="lift")])
    gaps = FailureAnalyser(report).to_gaps()
    d = gaps[0].to_dict()
    assert "failed_stage" in d and "suggested_generator" in d and "priority" in d


def test_summary_keys():
    report = _make_report([_ep(0, False, stage="lift")])
    s = FailureAnalyser(report).summary()
    assert "total_failures" in s and "failure_rate" in s and "n_clusters" in s
