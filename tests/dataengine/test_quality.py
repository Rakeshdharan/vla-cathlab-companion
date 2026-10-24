"""Data-quality rules (spec P6) in the shared generator and converter, for every engine."""
import json
from pathlib import Path

import numpy as np
import pytest

from contracts import StepInfo, TaskSpec, load_yaml
from dataengine.expert import ExpertParams, run_expert_episode
from dataengine.fake_sim import FakeExpertSim
from dataengine.generate import MemorySink, generate_scripted
from dataengine.npz_sink import NpzSink, merge_raw_shards, validate_shard
from dataengine.quality import is_noop
from dataengine.variation import VariationSpec, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


def test_is_noop_needs_no_motion_and_an_unchanged_gripper():
    still = np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32)
    assert is_noop(still, still) and is_noop(still, None)                    # LIBERO's definition
    assert not is_noop(still, np.array([0, 0, 0, 0, 0, 0, 1.0]))          # gripper toggles: keep
    assert not is_noop(np.array([0.01, 0, 0, 0, 0, 0, -1.0]), still)        # moves: keep


def test_generate_drops_noop_frames_and_records_provenance(tmp_path, monkeypatch):
    import dataengine.generate as gen
    real = gen.run_expert_episode

    def with_pause(sim, spec, cfg, params, q_down, record=None):
        def rec(obs, action):
            record(obs, action)
            if not hasattr(rec, "done"):
                rec.done = True
                for _ in range(3):
                    record(obs, np.array([0, 0, 0, 0, 0, 0, action[6]], dtype=np.float32))
        return real(sim, spec, cfg, params, q_down, record=rec)

    base = MemorySink()
    generate_scripted(FakeExpertSim(SPEC), SPEC, TASK, train_configs(SPEC)[:1], base, tmp_path / "base", "fake",
                      "cathlab/pick_fake", "v1", "test")
    base_meta = json.loads((tmp_path / "base/meta/episodes_extra.jsonl").read_text())
    monkeypatch.setattr(gen, "run_expert_episode", with_pause)
    sink = MemorySink()
    generate_scripted(FakeExpertSim(SPEC), SPEC, TASK, train_configs(SPEC)[:1], sink, tmp_path, "fake",
                      "cathlab/pick_fake", "v1", "test", extra_meta={"engine_version": "9.9"})
    (meta,) = [json.loads(line) for line in (tmp_path / "meta/episodes_extra.jsonl").read_text().splitlines()]
    assert meta["dropped_noops"] == base_meta["dropped_noops"] + 3 and meta["frames"] == base_meta["frames"]
    assert len(sink.episodes[0]) == meta["frames"]
    assert meta["provenance"]["engine_version"] == "9.9" and "git_commit" in meta["provenance"]

    sink = MemorySink()
    generate_scripted(FakeExpertSim(SPEC), SPEC, TASK, train_configs(SPEC)[:1], sink, tmp_path / "keep", "fake",
                      "cathlab/pick_fake", "v1", "test", drop_noops=False)
    kept = json.loads((tmp_path / "keep/meta/episodes_extra.jsonl").read_text())
    assert kept["dropped_noops"] == 0 and kept["frames"] == len(sink.episodes[0]) == kept["steps"] + 3


class _Flicker(FakeExpertSim):
    """Success drops out on its `drop_at`-th successful step, or on every `every`-th one."""

    def __init__(self, spec, drop_at=None, every=None):
        super().__init__(spec)
        self.drop_at, self.every = drop_at, every

    def reset(self, cfg, *a, **k):
        self.successes = 0
        return super().reset(cfg, *a, **k)

    def step(self, action):
        obs, info = super().step(action)
        if info.success:
            self.successes += 1
            if self.successes == self.drop_at or (self.every and self.successes % self.every == 0):
                info = StepInfo(stage_flags={**info.stage_flags, "place": False}, success=False)
        return obs, info


def test_success_must_hold():
    cfg = train_configs(SPEC)[0]
    clean = _Flicker(SPEC)
    assert run_expert_episode(clean, SPEC, cfg).success and clean.successes >= 10
    clean_steps = run_expert_episode(FakeExpertSim(SPEC), SPEC, cfg).steps
    late = run_expert_episode(_Flicker(SPEC, drop_at=clean.successes - 4), SPEC, cfg)
    assert late.success and late.steps > clean_steps          # the episode runs on until success holds again
    assert not run_expert_episode(_Flicker(SPEC, every=5), SPEC, cfg).success   # never holds 10 steps
    assert run_expert_episode(_Flicker(SPEC, every=5), SPEC, cfg, ExpertParams(success_hold=4)).success


def test_validate_shard_and_merge_catch_count_mismatches(tmp_path):
    sink = NpzSink(tmp_path)
    generate_scripted(FakeExpertSim(SPEC, image_size=32), SPEC, TASK, train_configs(SPEC)[:2], sink, tmp_path,
                      "fake", "cathlab/pick_fake", "v1", "test")
    assert validate_shard(tmp_path) == []
    (tmp_path / "episodes" / "episode_000001.npz").unlink()
    problems = validate_shard(tmp_path)
    assert any("episode_000001.npz" in p for p in problems) and any("2 in manifest" in p for p in problems)
    with pytest.raises(ValueError, match="episode_000001.npz"):
        list(merge_raw_shards([tmp_path]))
