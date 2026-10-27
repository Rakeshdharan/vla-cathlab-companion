"""
Failure analysis and GapSpec generation — A6 deliverable.

Clusters failed episodes into candidate GapSpecs that feed the Loop Controller.

Usage:
    from eval.failures import FailureAnalyser
    analyser = FailureAnalyser(report)
    gaps = analyser.to_gaps()
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts import EvalReport, EpisodeResult


@dataclass
class GapSpec:
    """A data gap identified from eval failures.

    Consumed by the Loop Controller (loop/) to request targeted synthetic data.
    """
    failed_stage: str
    objects: list[str]
    scene_variants: list[str]
    n_failures: int
    description: str
    suggested_generator: str  # "domain_rand" | "mimicgen" | "scripted" | "teleop"
    priority: float = 1.0  # higher = more urgent

    def to_dict(self) -> dict:
        return {
            "failed_stage": self.failed_stage,
            "objects": self.objects,
            "scene_variants": self.scene_variants,
            "n_failures": self.n_failures,
            "description": self.description,
            "suggested_generator": self.suggested_generator,
            "priority": self.priority,
        }


class FailureAnalyser:
    """Cluster failed episodes into GapSpecs.

    Clustering key: (failed_stage, object, scene_variant).
    Priority = n_failures / total_failures (relative frequency).
    Generator suggestion: simple heuristic based on stage.
    """

    # Heuristic: which generator is best for each failure stage
    _STAGE_TO_GENERATOR: dict[str, str] = {
        "reach": "domain_rand",
        "grasp": "scripted",
        "lift": "scripted",
        "transport": "mimicgen",
        "place": "mimicgen",
    }

    def __init__(self, report: "EvalReport"):
        self.report = report
        self._failed = [ep for ep in report.episodes if not ep.success]

    def cluster(self) -> dict[tuple, list["EpisodeResult"]]:
        """Group failures by (failed_stage, object, scene_variant)."""
        groups: dict[tuple, list] = defaultdict(list)
        for ep in self._failed:
            key = (ep.failed_stage or "unknown", ep.object, ep.scene_variant)
            groups[key].append(ep)
        return dict(groups)

    def to_gaps(self, min_failures: int = 1) -> list[GapSpec]:
        """Convert clusters to GapSpecs, sorted by priority (highest first).

        Args:
            min_failures: Minimum failures in a cluster to produce a GapSpec.

        Returns:
            List of GapSpec sorted by priority descending.
        """
        total = max(len(self._failed), 1)
        clusters = self.cluster()
        gaps = []

        for (stage, obj, variant), eps in clusters.items():
            n = len(eps)
            if n < min_failures:
                continue
            priority = n / total
            generator = self._STAGE_TO_GENERATOR.get(stage, "domain_rand")
            desc = (
                f"{n} failures at '{stage}' stage for object '{obj}' "
                f"in scene variant '{variant}'. "
                f"Recommended generator: {generator}."
            )
            gaps.append(GapSpec(
                failed_stage=stage,
                objects=[obj],
                scene_variants=[variant],
                n_failures=n,
                description=desc,
                suggested_generator=generator,
                priority=priority,
            ))

        # Merge gaps with same stage+generator across objects/variants
        gaps.sort(key=lambda g: g.priority, reverse=True)
        return gaps

    def summary(self) -> dict:
        """Return a summary dict for logging."""
        clusters = self.cluster()
        return {
            "total_failures": len(self._failed),
            "total_episodes": len(self.report.episodes),
            "failure_rate": len(self._failed) / max(len(self.report.episodes), 1),
            "n_clusters": len(clusters),
            "top_cluster": max(
                ((k, len(v)) for k, v in clusters.items()),
                key=lambda x: x[1],
                default=(None, 0),
            )[0],
        }
