import json
from pathlib import Path
from contracts import DatasetManifest, TaskSpec, load_yaml
from dataengine.fake_sim import FakeExpertSim
from dataengine.generate import MemorySink, generate_scripted, instruction_for
from dataengine.variation import VariationSpec, train_configs

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


def test_generate_records_successful_episodes_with_metadata(tmp_path):
    cfgs = train_configs(SPEC)[:6]
    sink = MemorySink()
    report, manifest = generate_scripted(FakeExpertSim(SPEC), SPEC, TASK, cfgs, sink, tmp_path, "fake",
                                         "cathlab/pick_fake", "v1", "test")
    assert report.attempted == 6 and report.succeeded == 6 and sink.finalized
    assert len(sink.episodes) == 6
    frame = sink.episodes[0][0]
    assert set(frame) == {"observation.state", "observation.images.agentview", "observation.images.eye_in_hand",
                          "action", "task"}
    assert frame["action"].shape == (7,) and frame["observation.state"].shape == (8,)
    assert frame["task"] == instruction_for(TASK, cfgs[0])
    lines = [json.loads(l) for l in (tmp_path / "meta/episodes_extra.jsonl").read_text().splitlines()]
    assert [l["episode_index"] for l in lines] == list(range(6)) and lines[0]["source_sim"] == "fake"
    assert load_yaml(tmp_path / "manifest.yaml", DatasetManifest) == manifest
    assert manifest.source == "scripted" and manifest.n_episodes == 6


class _FailingSim(FakeExpertSim):
    def tray_contains(self, name):
        return False


def test_failed_episodes_are_discarded(tmp_path):
    sink = MemorySink()
    report, manifest = generate_scripted(_FailingSim(SPEC), SPEC, TASK, train_configs(SPEC)[:2], sink, tmp_path,
                                         "fake", "cathlab/pick_fake", "v1", "test")
    assert report.succeeded == 0 and sink.episodes == [] and manifest.n_episodes == 0
