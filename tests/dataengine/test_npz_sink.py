import json
from pathlib import Path
import numpy as np
import pytest
from contracts import TaskSpec, load_yaml
from dataengine.fake_sim import FakeExpertSim
from dataengine.generate import generate_scripted
from dataengine.npz_sink import NpzSink, iter_npz_frames, merge_raw_shards
from dataengine.variation import VariationSpec, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


class _FailSecond(FakeExpertSim):
    calls = 0

    def reset(self, cfg, *a, **k):
        type(self).calls += 1
        self.fail = type(self).calls == 2
        return super().reset(cfg, *a, **k)

    def tray_contains(self, name):
        return not self.fail and super().tray_contains(name)


def _generate(out, cfgs, sim=None):
    sink = NpzSink(out)
    report, _ = generate_scripted(sim or FakeExpertSim(SPEC, image_size=32), SPEC, TASK, cfgs, sink, out, "fake",
                                  "cathlab/pick_fake", "v1", "test")
    return sink, report


def test_npz_sink_writes_one_file_per_kept_episode(tmp_path):
    _FailSecond.calls = 0
    sink, report = _generate(tmp_path, train_configs(SPEC)[:3], _FailSecond(SPEC, image_size=32))
    assert report.succeeded == 2 and sink.finalized
    files = sorted((tmp_path / "episodes").glob("*.npz"))
    assert [f.name for f in files] == ["episode_000000.npz", "episode_000001.npz"]
    frames = list(iter_npz_frames(files[0]))
    meta = [json.loads(l) for l in (tmp_path / "meta/episodes_extra.jsonl").read_text().splitlines()]
    assert len(frames) == meta[0]["frames"] == meta[0]["steps"] - meta[0]["dropped_noops"]
    assert frames[0]["task"] == meta[0]["instruction"]
    assert frames[0]["observation.state"].shape == (8,) and frames[0]["action"].dtype == np.float32
    assert frames[0]["observation.images.agentview"].shape == (32, 32, 3)


def test_npz_sink_refuses_to_overwrite(tmp_path):
    _generate(tmp_path, train_configs(SPEC)[:1])
    with pytest.raises(FileExistsError):
        NpzSink(tmp_path)


def test_merge_raw_shards_reindexes_in_shard_order(tmp_path):
    cfgs = train_configs(SPEC)
    _generate(tmp_path / "shard_001", cfgs[3:4])
    _generate(tmp_path / "shard_000", cfgs[0:2])
    merged = list(merge_raw_shards([tmp_path / "shard_000", tmp_path / "shard_001"]))
    assert [m["episode_index"] for m, _ in merged] == [0, 1, 2]
    assert [m["config"]["seed"] for m, _ in merged] == [0, 1, 3]
    assert merged[2][0]["raw_shard"] == "shard_001" and merged[2][1].name == "episode_000000.npz"
