import pytest
from orchestrator.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "var" / "control.db")


def test_runs_roundtrip(store):
    run = store.create_run({"simulator": "robosuite", "objects": ["vial"]}, note="first")
    assert run.id.startswith("run-")
    assert store.get_run(run.id) == run
    assert [r.id for r in store.list_runs()] == [run.id]
    assert store.get_run("nope") is None


def test_job_lifecycle_merges_results(store):
    job = store.create_job("stage", "generate", params={"n": 3}, run_id="run-1")
    assert job.state == "queued" and job.params == {"n": 3}
    store.update_job(job.id, state="running", worker="robosuite", result={"episodes": 1})
    done = store.update_job(job.id, state="done", result={"success": 1.0})
    assert done.state == "done" and done.worker == "robosuite"
    assert done.result == {"episodes": 1, "success": 1.0}
    assert done.updated >= done.created


def test_list_jobs_filters_newest_first(store):
    a = store.create_job("stage", "generate", run_id="r1")
    b = store.create_job("agent", "add_object")
    c = store.create_job("stage", "eval", run_id="r2")
    assert [j.id for j in store.list_jobs()] == [c.id, b.id, a.id]
    assert [j.id for j in store.list_jobs(kind="stage")] == [c.id, a.id]
    assert [j.id for j in store.list_jobs(run_id="r1")] == [a.id]


def test_update_unknown_job_raises(store):
    with pytest.raises(KeyError):
        store.update_job("missing", state="done")
