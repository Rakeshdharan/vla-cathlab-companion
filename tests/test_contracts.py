from pathlib import Path
import pytest
import numpy as np
from pydantic import ValidationError
from contracts import (
    STAGES, TaskSpec, SceneSpec, DatasetManifest, load_yaml, save_yaml,
    RandomPolicy, DummySim, StepInfo,
    EpisodeResult, EvalReport, ModelCard,
)

ROOT = Path(__file__).resolve().parents[1]


def test_stages_fixed():
    assert STAGES == ("reach", "grasp", "lift", "transport", "place")


def test_task_spec_loads_and_formats():
    task = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)
    assert "guidewire" in task.objects
    assert "hand me the guidewire" in [s.lower() for s in task.instructions("guidewire")]


def test_task_spec_rejects_unknown_stage():
    with pytest.raises(ValidationError):
        TaskSpec(task_id="t", instruction_templates=["x {object}"], objects=["a"],
                 success_predicate="p", stage_predicates={"fly": "p"})


def test_scene_spec_loads():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    assert scene.backend == "dummy"


def test_manifest_roundtrip(tmp_path):
    m = DatasetManifest(dataset_id="cathlab/handover", version="v0", source="provided",
                        task_id="handover", scene_id="cathlab_v1", n_episodes=500, created_by="rakesh")
    p = tmp_path / "manifest.yaml"
    save_yaml(m, p)
    assert load_yaml(p, DatasetManifest) == m


def test_manifest_rejects_unknown_source():
    with pytest.raises(ValidationError):
        DatasetManifest(dataset_id="d", version="v0", source="magic", task_id="t",
                        scene_id="s", n_episodes=1, created_by="x")


def test_random_policy_shape():
    p = RandomPolicy(action_dim=16, chunk_size=10)
    p.load("none")
    a = p.predict({"state": np.zeros(16)}, "hand me the guidewire")
    assert a.shape == (10, 16)
    assert p.meta().action_dim == 16


def test_dummy_sim_progresses_through_stages():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    sim = DummySim(steps_per_stage=2)
    obs = sim.reset(scene, seed=0, target_object="guidewire")
    assert set(obs) >= {"state", "image_head"}
    info = None
    for _ in range(10):
        obs, info = sim.step(np.zeros(16))
    assert isinstance(info, StepInfo) and info.success
    assert all(info.stage_flags[s] for s in STAGES)


def test_dummy_sim_can_fail_at_stage():
    scene = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
    sim = DummySim(steps_per_stage=1, fail_at="grasp")
    sim.reset(scene, seed=0, target_object="sheath")
    for _ in range(20):
        _, info = sim.step(np.zeros(16))
    assert not info.success and info.stage_flags["reach"] and not info.stage_flags["grasp"]


def _ep(i, ok, stage=None):
    return EpisodeResult(episode=i, seed=i, object="guidewire", instruction="hand me the guidewire",
                         success=ok, failed_stage=stage, steps=10, latency_ms_p50=20.0, latency_ms_p95=40.0)


def test_eval_report_metrics(tmp_path):
    r = EvalReport(model_version="smolvla@v0", dataset_versions=["cathlab/handover@v0"], backend="dummy",
                   episodes=[_ep(0, True), _ep(1, False, "grasp"), _ep(2, False, "grasp"), _ep(3, True)])
    assert r.success_rate() == 0.5
    lo, hi = r.wilson_ci()
    assert 0.0 < lo < 0.5 < hi < 1.0
    assert r.stage_failure_hist() == {"reach": 0, "grasp": 2, "lift": 0, "transport": 0, "place": 0}
    r.save(tmp_path / "r.json")
    assert EvalReport.load(tmp_path / "r.json") == r


def test_model_card_roundtrip(tmp_path):
    c = ModelCard(model_name="cathlab-pick", version="v0", adapter="smolvla", ckpt_uri="hf://x",
                  dataset_versions=["cathlab/handover@v0"], eval_report_path="reports/v0.json")
    save_yaml(c, tmp_path / "card.yaml")
    assert load_yaml(tmp_path / "card.yaml", ModelCard) == c
