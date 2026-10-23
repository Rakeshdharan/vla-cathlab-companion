"""Closed-loop eval and live rollout in robosuite (run in the `robosuite` env). robosuite is imported lazily."""
import json
import time
from pathlib import Path
import numpy as np
import yaml
from contracts import STAGES, EpisodeResult, EvalReport, TaskSpec, load_yaml
from dataengine.variation import TIERS, VariationSpec

VIDEO_MODES = ("none", "failures", "all")
FPS = 20
FRAME_EVERY = 2          # keep every 2nd frame in videos


def load_suite(path) -> dict:
    suite = yaml.safe_load(Path(path).read_text())
    bad = [t for t in suite["tiers"] if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tiers {bad} in {path}; expected {list(TIERS)}")
    return suite


def _backend(spec: VariationSpec, tier: str):
    from sim.robosuite.backend import RobosuiteBackend   # lazy: needs robosuite

    class _Backend(RobosuiteBackend):
        on_reset = None      # ExpertPolicy.begin when the expert is evaluated

        def reset_config(self, cfg):
            obs = super().reset_config(cfg)
            if self.on_reset is not None:
                self.on_reset(cfg)
            return obs

    return _Backend(spec, tier=tier)


def _policy(policy_arg: str, backend, spec: VariationSpec):
    if policy_arg == "expert":
        from engines.expert_policy import ExpertPolicy
        from sim.robosuite.params import ROBOSUITE_EXPERT_PARAMS
        policy = ExpertPolicy(backend.sim, spec, params=ROBOSUITE_EXPERT_PARAMS, control_hz=FPS)
        backend.on_reset = policy.begin
    else:
        from engines.policies import make_policy
        policy = make_policy(policy_arg, action_dim=7)
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
        chunk = np.asarray(policy.predict(obs, instruction), dtype=np.float32).reshape(-1, 7)
        latencies.append((time.perf_counter() - t0) * 1000)
        for action in chunk:
            obs, info = backend.step(action)
            steps += 1
            if frames is not None and steps % FRAME_EVERY == 0:
                frames.append(np.concatenate([obs["agentview"], obs["eye_in_hand"]], axis=1))
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


def save_video(frames: list, stem: Path, fps: float) -> Path:
    import imageio.v2 as imageio
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio_ffmpeg  # noqa: F401
        path = stem.with_suffix(".mp4")
        imageio.mimsave(path, frames, fps=fps)
    except ImportError:
        path = stem.with_suffix(".gif")
        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
    return path


def run_eval(suite_path, policy_arg: str, out: Path, spec_path, task_path, model_version: str | None = None,
             objects: list[str] | None = None, tiers: list[str] | None = None, max_seeds: int | None = None,
             video: str = "failures") -> EvalReport:
    if video not in VIDEO_MODES:
        raise ValueError(f"--video must be one of {VIDEO_MODES}")
    suite = load_suite(suite_path)
    spec, task = VariationSpec.load(spec_path), load_yaml(task_path, TaskSpec)
    objects = objects or suite.get("objects") or task.objects
    tiers = tiers or suite["tiers"]
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise ValueError(f"unknown tiers {bad}; expected {list(TIERS)}")
    seeds = suite["seeds"][:max_seeds] if max_seeds else suite["seeds"]
    out = Path(out)
    backend = _backend(spec, tiers[0])
    episodes: list[EpisodeResult] = []
    try:
        policy = _policy(policy_arg, backend, spec)
        for tier in tiers:
            backend.tier = tier
            for obj in objects:
                for seed in seeds:
                    frames = None if video == "none" else []
                    ep = run_episode(policy, backend, task, obj, seed, suite["max_steps"], frames=frames)
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
    report = EvalReport(model_version=model_version or policy_arg, dataset_versions=[], backend="robosuite",
                        episodes=episodes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval_report.json").write_text(report.model_dump_json(indent=2))
    lo, hi = report.wilson_ci()
    print(f"success {report.success_rate():.0%} (95% CI {lo:.0%}-{hi:.0%}) over {len(episodes)} episodes", flush=True)
    return report


def run_rollout(policy_arg: str, out: Path, spec_path, task_path, obj: str, seed: int = 5000,
                tier: str = "in_distribution", instruction: str | None = None, max_steps: int = 400) -> dict:
    spec, task = VariationSpec.load(spec_path), load_yaml(task_path, TaskSpec)
    out = Path(out)
    backend = _backend(spec, tier)
    frames: list = []
    try:
        policy = _policy(policy_arg, backend, spec)
        ep = run_episode(policy, backend, task, obj, seed, max_steps, instruction=instruction, frames=frames)
    finally:
        backend.close()
    video = save_video(frames, out / "rollout", FPS / FRAME_EVERY) if frames else None
    result = {"success": ep.success, "steps": ep.steps, "failed_stage": ep.failed_stage,
              "latency_ms_p50": ep.latency_ms_p50, "latency_ms_p95": ep.latency_ms_p95, "object": obj,
              "tier": tier, "seed": seed, "instruction": ep.instruction, "policy": policy_arg,
              "video": video.name if video else None}
    out.mkdir(parents=True, exist_ok=True)
    (out / "rollout.json").write_text(json.dumps(result, indent=2))
    print(f"rollout {obj} ({ep.instruction!r}): {'success' if ep.success else 'failed at ' + str(ep.failed_stage)} "
          f"in {ep.steps} steps", flush=True)
    return result
