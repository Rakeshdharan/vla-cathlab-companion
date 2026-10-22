"""
Evaluation runner — core loop.

Public API (signatures FROZEN per spec §4 / akanksh.md):
    run_episode(policy, sim, scene, task, obj, seed, max_steps=200) -> EpisodeResult
    run_suite(policy, sim, scene, task, seeds, model_version, dataset_versions) -> EvalReport

Akanksh's extensions (metrics, video, VLM judge, tiers) live in
eval/metrics.py, eval/judge.py, eval/cli.py, eval/benchmark.py, eval/failures.py.
They call run_episode/run_suite; they do NOT modify these functions.
"""

import time
import numpy as np
from contracts import (
    STAGES,
    EpisodeResult,
    EvalReport,
    PolicyAdapter,
    SceneSpec,
    SimBackend,
    TaskSpec,
)


def run_episode(
    policy: PolicyAdapter,
    sim: SimBackend,
    scene: SceneSpec,
    task: TaskSpec,
    obj: str,
    seed: int,
    max_steps: int = 200,
) -> EpisodeResult:
    """Run one episode and return a fully populated EpisodeResult.

    Args:
        policy: Any PolicyAdapter implementation.
        sim: Any SimBackend implementation.
        scene: SceneSpec for this episode.
        task: TaskSpec defining the task.
        obj: Target object name (must be in task.objects).
        seed: Integer seed passed to sim.reset for reproducibility.
        max_steps: Hard step limit before declaring failure.

    Returns:
        EpisodeResult with success, failed_stage, latency stats, and step count.
    """
    instruction = task.instructions(obj)[seed % len(task.instruction_templates)]
    obs = sim.reset(scene, seed=seed, target_object=obj)
    latencies: list[float] = []
    steps = 0
    info = None

    while steps < max_steps:
        t0 = time.perf_counter()
        chunk = policy.predict(obs, instruction)
        latencies.append((time.perf_counter() - t0) * 1000)

        for action in chunk:
            obs, info = sim.step(action)
            steps += 1
            if info.success or steps >= max_steps:
                break
        if info.success:
            break

    # Determine failed stage: first stage not completed
    failed = None if info.success else next(
        (s for s in STAGES if not info.stage_flags.get(s, False)), STAGES[0]
    )

    return EpisodeResult(
        episode=seed,
        seed=seed,
        object=obj,
        instruction=instruction,
        success=info.success,
        failed_stage=failed,
        steps=steps,
        latency_ms_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
        latency_ms_p95=float(np.percentile(latencies, 95)) if latencies else 0.0,
    )


def run_suite(
    policy: PolicyAdapter,
    sim: SimBackend,
    scene: SceneSpec,
    task: TaskSpec,
    seeds: list[int],
    model_version: str,
    dataset_versions: list[str],
) -> EvalReport:
    """Run all objects × all seeds and return an EvalReport.

    Args:
        policy: Any PolicyAdapter implementation.
        sim: Any SimBackend implementation.
        scene: SceneSpec to use for all episodes.
        task: TaskSpec defining the task and object set.
        seeds: List of integer seeds; each seed runs all objects.
        model_version: Version string for the report (e.g. "smolvla@v0").
        dataset_versions: Dataset version strings consumed by this model.

    Returns:
        EvalReport aggregating all episodes.
    """
    episodes = []
    for obj in task.objects:
        for seed in seeds:
            ep = run_episode(policy, sim, scene, task, obj, seed)
            episodes.append(ep.model_copy(update={"episode": len(episodes)}))
    return EvalReport(
        model_version=model_version,
        dataset_versions=dataset_versions,
        backend=scene.backend,
        episodes=episodes,
    )
