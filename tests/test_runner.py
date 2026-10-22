from pathlib import Path
from contracts import DummySim, RandomPolicy, SceneSpec, TaskSpec, load_yaml
from eval.runner import run_episode, run_suite

ROOT = Path(__file__).resolve().parents[1]
SCENE = load_yaml(ROOT / "configs/scenes/dummy.yaml", SceneSpec)
TASK = load_yaml(ROOT / "configs/tasks/handover.yaml", TaskSpec)


def test_run_episode_success():
    r = run_episode(RandomPolicy(), DummySim(steps_per_stage=2), SCENE, TASK, "guidewire", seed=0)
    assert r.success and r.failed_stage is None and r.latency_ms_p95 >= r.latency_ms_p50 >= 0


def test_run_episode_reports_failed_stage():
    r = run_episode(RandomPolicy(), DummySim(steps_per_stage=1, fail_at="lift"), SCENE, TASK, "sheath",
                    seed=1, max_steps=30)
    assert not r.success and r.failed_stage == "lift"


def test_run_suite_aggregates():
    rep = run_suite(RandomPolicy(), DummySim(steps_per_stage=1), SCENE, TASK, seeds=[0, 1, 2],
                    model_version="random@v0", dataset_versions=[])
    assert len(rep.episodes) == 3 * len(TASK.objects) and rep.success_rate() == 1.0
