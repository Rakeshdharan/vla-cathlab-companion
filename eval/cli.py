"""
Eval CLI — A3 deliverable.

Usage:
    python -m eval.cli --policy <adapter|url> --sim <backend> --suite suite_v1 [options]

Extends run_suite without changing its signature by wrapping it with:
  - per-tier scene variant selection
  - video capture toggle
  - VLM judge attachment
  - W&B logging
  - EvalReport JSON output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_suite_config(suite_name: str) -> dict:
    """Load eval suite YAML config."""
    import yaml
    path = ROOT / "configs" / "eval" / f"{suite_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _make_policy(policy_arg: str):
    """Resolve --policy to a PolicyAdapter.

    Supports:
      - 'random'           → RandomPolicy
      - 'ws://...'         → PolicyClient (Sharan's serve package)
      - 'path/to/adapter'  → dynamic import (not yet implemented)
    """
    from contracts import RandomPolicy
    if policy_arg == "random":
        return RandomPolicy()
    # Lazy import PolicyClient to avoid mandatory dependency on serve/
    if policy_arg.startswith("ws://") or policy_arg.startswith("wss://"):
        try:
            from serve.client import PolicyClient  # type: ignore
            return PolicyClient(policy_arg)
        except ImportError:
            print("ERROR: serve.client not available. Install the 'serve' package.", file=sys.stderr)
            sys.exit(1)
    print(f"ERROR: Unknown --policy '{policy_arg}'. Use 'random' or a WebSocket URL.", file=sys.stderr)
    sys.exit(1)


def _make_sim(sim_arg: str, scene):
    """Resolve --sim to a SimBackend."""
    from contracts import DummySim
    if sim_arg == "dummy":
        return DummySim()
    if sim_arg == "isaac":
        try:
            from sim.isaac import IsaacBackend  # type: ignore
            return IsaacBackend()
        except ImportError:
            print("WARNING: IsaacBackend not available, falling back to DummySim.", file=sys.stderr)
            return DummySim()
    if sim_arg == "robosimstudio":
        try:
            from sim.robosimstudio import RoboSimStudioBackend  # type: ignore
            return RoboSimStudioBackend()
        except ImportError:
            print("WARNING: RoboSimStudioBackend not available, falling back to DummySim.", file=sys.stderr)
            return DummySim()
    print(f"ERROR: Unknown --sim '{sim_arg}'.", file=sys.stderr)
    sys.exit(1)


def _attach_judge_verdicts(episodes, judge, use_vlm: bool):
    """Annotate episodes in-place with judge verdicts."""
    updated = []
    for ep in episodes:
        verdict = judge.judge(
            video_path=ep.video_path,
            instruction=ep.instruction,
            ground_truth_success=ep.success,
            ground_truth_failed_stage=ep.failed_stage,
        )
        updated.append(ep.model_copy(update={"judge_verdict": json.dumps(verdict.to_dict())}))
    return updated


def run_cli(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="CathLab eval harness")
    parser.add_argument("--policy", required=True, help="Policy: 'random', ws://..., or adapter path")
    parser.add_argument("--sim", default="dummy", help="Simulator backend: dummy|isaac|robosimstudio")
    parser.add_argument("--suite", default="suite_v1", help="Eval suite config name (without .yaml)")
    parser.add_argument("--model-version", default="unknown@v0", help="Model version string for report")
    parser.add_argument("--dataset-versions", nargs="*", default=[], help="Dataset version strings")
    parser.add_argument("--out", default="reports", help="Output directory for EvalReport JSON")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--vlm-judge", action="store_true", help="Run VLM success judge on videos")
    parser.add_argument("--tier", default=None, help="Run only this difficulty tier")
    ns = parser.parse_args(args)

    # Load suite config
    suite_cfg = _load_suite_config(ns.suite)
    from contracts import load_yaml, SceneSpec, TaskSpec
    task_cfg_path = ROOT / "configs" / "tasks" / f"{suite_cfg['task']}.yaml"
    task = load_yaml(task_cfg_path, TaskSpec)

    # Load scene spec (use dummy scene for dummy sim)
    if ns.sim == "dummy":
        scene_path = ROOT / "configs" / "scenes" / "dummy.yaml"
    else:
        scene_path = ROOT / "configs" / "scenes" / "cathlab_v1.yaml"
    scene = load_yaml(scene_path, SceneSpec)

    policy = _make_policy(ns.policy)
    sim = _make_sim(ns.sim, scene)

    # Judge
    from eval.judge import get_judge
    judge = get_judge(use_vlm=ns.vlm_judge)

    # Run each difficulty tier
    from eval.runner import run_suite
    from contracts import EvalReport

    all_episodes = []
    tiers = suite_cfg["difficulty_tiers"]
    if ns.tier:
        tiers = [t for t in tiers if t["name"] == ns.tier]

    for tier in tiers:
        tier_name = tier["name"]
        seeds = suite_cfg["seeds"].get(tier_name, suite_cfg["seeds"]["in_distribution"])
        print(f"  Running tier '{tier_name}' ({len(seeds)} seeds × {len(task.objects)} objects) …")

        report = run_suite(
            policy=policy,
            sim=sim,
            scene=scene,
            task=task,
            seeds=seeds,
            model_version=ns.model_version,
            dataset_versions=ns.dataset_versions,
        )

        # Tag episodes with scene_variant / tier
        tier_eps = [ep.model_copy(update={"scene_variant": tier_name}) for ep in report.episodes]

        # Attach judge verdicts
        tier_eps = _attach_judge_verdicts(tier_eps, judge, ns.vlm_judge)
        all_episodes.extend(tier_eps)

    # Build final report
    final_report = EvalReport(
        model_version=ns.model_version,
        dataset_versions=ns.dataset_versions,
        backend=scene.backend,
        episodes=all_episodes,
    )

    # Save
    out_dir = Path(ns.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ns.model_version.replace('/', '_').replace('@', '-')}.json"
    final_report.save(out_path)
    print(f"\nEvalReport saved → {out_path}")
    print(f"Success rate: {final_report.success_rate():.2%}  CI: {final_report.wilson_ci()}")
    print(f"Stage failures: {final_report.stage_failure_hist()}")

    # W&B logging
    if ns.wandb:
        try:
            import wandb  # type: ignore
            from eval.metrics import full_summary
            run = wandb.init(project="cathlab-companion", name=ns.model_version)
            run.log(full_summary(final_report))
            run.log_artifact(str(out_path), name="eval_report", type="eval")
            run.finish()
            print(f"W&B run logged: {run.url}")
        except ImportError:
            print("WARNING: wandb not installed, skipping W&B logging.")


if __name__ == "__main__":
    run_cli()
