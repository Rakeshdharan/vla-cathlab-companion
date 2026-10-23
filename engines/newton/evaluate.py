"""Closed-loop eval and live rollout in Newton, FR3 Duo (run in ~/venvs/newton). newton is imported lazily.

Same suite, tiers, seeds and EvalReport as engines/robosuite/evaluate.py; the episode loop is the FR3 one
(16-D joint targets at 30 Hz, three 224² cameras), so `max_steps` from the suite (20 Hz frames) is scaled
by 30/20 to keep the same time budget.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
from contracts import STAGES, EpisodeResult, EvalReport, TaskSpec, load_yaml
from engines.robosuite.evaluate import VIDEO_MODES, load_suite, save_video

BACKEND = "newton"
FPS = 30
FRAME_EVERY = 2          # keep every 2nd frame in videos
ACTION_DIM = 16
SUITE_HZ = 20


def frames_for(suite_max_steps: int) -> int:
    return math.ceil(suite_max_steps * FPS / SUITE_HZ)


def _backend(spec_path, tier: str):
    from sim.newton.backend import NewtonFR3Backend   # lazy: needs newton
    return NewtonFR3Backend(spec_path, tier=tier)


def _policy(policy_arg: str, backend):
    if policy_arg == "expert":
        from sim.newton.expert_policy import FR3ExpertPolicy
        policy = FR3ExpertPolicy(backend.variation)
        backend.on_reset = policy.begin
    else:
        from engines.policies import make_policy
        policy = make_policy(policy_arg, action_dim=ACTION_DIM)
        backend.on_reset = None
    policy.load("")
    return policy


def run_episode(policy, backend, task: TaskSpec, obj: str, seed: int, max_steps: int,
                instruction: str | None = None, frames: list | None = None) -> EpisodeResult:
    instruction = instruction or task.instructions(obj)[seed % len(task.instruction_templates)]
    obs = backend.reset(None, seed=seed, target_object=obj)
    latencies: list[float] = []
    steps, info = 0, None
    while steps < max_steps:
        t0 = time.perf_counter()
        chunk = np.asarray(policy.predict(obs, instruction), dtype=np.float32).reshape(-1, ACTION_DIM)
        latencies.append((time.perf_counter() - t0) * 1000)
        for action in chunk:
            obs, info = backend.step(action)
            steps += 1
            if frames is not None and steps % FRAME_EVERY == 0:
                frames.append(np.concatenate([obs["external_camera"], obs["right_wrist_camera"]], axis=1))
            if info.success or steps >= max_steps:
                break
        if info.success:
            break
    success = bool(info is not None and info.success)
    failed = None if success else next((s for s in STAGES if not (info and info.stage_flags.get(s))), STAGES[0])
    return EpisodeResult(episode=0, seed=seed, object=obj, instruction=instruction, success=success,
                         failed_stage=failed, steps=steps,
                         latency_ms_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
                         latency_ms_p95=float(np.percentile(latencies, 95)) if latencies else 0.0)


def run_eval(suite_path, policy_arg: str, out: Path, spec_path, task_path, model_version: str | None = None,
             objects: list[str] | None = None, tiers: list[str] | None = None, max_seeds: int | None = None,
             video: str = "failures") -> EvalReport:
    if video not in VIDEO_MODES:
        raise ValueError(f"--video must be one of {VIDEO_MODES}")
    suite = load_suite(suite_path)
    task = load_yaml(task_path, TaskSpec)
    objects = objects or suite.get("objects") or task.objects
    tiers = tiers or suite["tiers"]
    seeds = suite["seeds"][:max_seeds] if max_seeds else suite["seeds"]
    max_steps = frames_for(suite["max_steps"])
    out = Path(out)
    backend = _backend(spec_path, tiers[0])
    episodes: list[EpisodeResult] = []
    try:
        policy = _policy(policy_arg, backend)
        for tier in tiers:
            backend.tier = tier
            for obj in objects:
                for seed in seeds:
                    frames = None if video == "none" else []
                    ep = run_episode(policy, backend, task, obj, seed, max_steps, frames=frames)
                    ep.episode, ep.scene_variant = len(episodes), tier
                    if frames and (video == "all" or not ep.success):
                        path = save_video(frames, out / "videos" / f"{tier}_{obj}_{seed}", FPS / FRAME_EVERY)
                        ep.video_path = str(path.relative_to(out))
                    episodes.append(ep)
                    verdict = "OK  " if ep.success else f"FAIL at {ep.failed_stage}"
                    print(f"[{len(episodes)}] {tier:22s} {obj:8s} seed {seed}: {verdict} ({ep.steps} steps)",
                          flush=True)
    finally:
        backend.close()
    report = EvalReport(model_version=model_version or policy_arg, dataset_versions=[], backend=BACKEND,
                        episodes=episodes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(report.model_dump_json(indent=2))
    lo, hi = report.wilson_ci()
    print(f"success {report.success_rate():.0%} (95% CI {lo:.0%}-{hi:.0%}) over {len(episodes)} episodes", flush=True)
    return report


def run_rollout(policy_arg: str, out: Path, spec_path, task_path, obj: str, seed: int = 5000,
                tier: str = "in_distribution", instruction: str | None = None, max_steps: int = 600) -> dict:
    task = load_yaml(task_path, TaskSpec)
    out = Path(out)
    backend = _backend(spec_path, tier)
    frames: list = []
    try:
        policy = _policy(policy_arg, backend)
        ep = run_episode(policy, backend, task, obj, seed, max_steps, instruction=instruction, frames=frames)
    finally:
        backend.close()
    video = save_video(frames, out / "rollout", FPS / FRAME_EVERY) if frames else None
    result = {"success": ep.success, "steps": ep.steps, "failed_stage": ep.failed_stage,
              "latency_ms_p50": ep.latency_ms_p50, "latency_ms_p95": ep.latency_ms_p95, "object": obj,
              "tier": tier, "seed": seed, "instruction": ep.instruction, "policy": policy_arg, "engine": BACKEND,
              "video": video.name if video else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout.json").write_text(json.dumps(result, indent=2))
    print(f"rollout {obj} ({ep.instruction!r}): {'success' if ep.success else 'failed at ' + str(ep.failed_stage)} "
          f"in {ep.steps} steps", flush=True)
    return result
