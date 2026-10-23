import threading
from pathlib import Path
import uvicorn
from fastapi.testclient import TestClient
from orchestrator.control import create_control_app
from orchestrator.worker import create_worker_app

ROOT = Path(__file__).resolve().parents[2]


def client(tmp_path, workers_yaml: str = "workers: []\n", frontend=True, harness=None):
    wp = tmp_path / "workers.yaml"
    wp.write_text(workers_yaml)
    fe = tmp_path / "fe"
    if frontend:
        fe.mkdir()
        (fe / "index.html").write_text("<title>CathLab Control Station</title>")
    app = create_control_app(ROOT, var_dir=tmp_path / "var", workers_path=wp, frontend_dir=fe, harness=harness)
    return TestClient(app)


def test_catalog_and_selection_check(tmp_path):
    c = client(tmp_path)
    cat = c.get("/api/catalog").json()
    assert "simulator" in cat["sections"] and cat["defaults"]["model"] == "smolvla"
    ok = c.post("/api/selection/check", json={"selection": cat["defaults"]}).json()
    assert ok["problems"] == []
    bad = c.post("/api/selection/check", json={"selection": {**cat["defaults"], "simulator": "isaac"}}).json()
    assert any(p["harness_task"] == "add_sim_backend" for p in bad["problems"])


def test_runs_are_frozen_to_disk_and_blocked_when_not_runnable(tmp_path):
    c = client(tmp_path)
    defaults = c.get("/api/catalog").json()["defaults"]
    run = c.post("/api/runs", json={"selection": defaults, "note": "smoke"}).json()
    assert (tmp_path / "var/runs" / run["id"] / "run.yaml").exists()
    assert c.get(f"/api/runs/{run['id']}").json()["note"] == "smoke"
    assert [r["id"] for r in c.get("/api/runs").json()] == [run["id"]]
    r = c.post("/api/runs", json={"selection": {**defaults, "expert": "graspgen"}})
    assert r.status_code == 409 and r.json()["detail"]["problems"][0]["harness_task"] == "add_expert"
    assert c.get("/api/runs/nope").status_code == 404


def test_jobs_endpoints(tmp_path):
    c = client(tmp_path)
    store = c.app.state.store
    j = store.create_job("stage", "generate", run_id="run-x")
    assert [x["id"] for x in c.get("/api/jobs", params={"run_id": "run-x"}).json()] == [j.id]
    assert c.get(f"/api/jobs/{j.id}").json()["name"] == "generate"
    assert c.get("/api/jobs/nope").status_code == 404


def test_workers_health_online_and_offline(tmp_path):
    app = create_worker_app("robosuite", ["generate"], tmp_path / "w")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8799, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        pass
    try:
        c = client(tmp_path, "workers:\n  - {name: robosuite, url: 'http://127.0.0.1:8799'}\n"
                             "  - {name: lerobot, url: 'http://127.0.0.1:8798'}\n")
        ws = {w["name"]: w for w in c.get("/api/workers").json()}
        assert ws["robosuite"]["healthy"] and ws["robosuite"]["capabilities"] == ["generate"]
        assert not ws["lerobot"]["healthy"] and ws["lerobot"]["error"]
    finally:
        server.should_exit = True
        t.join(5)


def test_frontend_served_and_agent_optional(tmp_path):
    c = client(tmp_path)
    assert "CathLab Control Station" in c.get("/").text
    assert c.get("/api/health").json()["agent"] is False
    assert c.get("/api/agent/tasks").status_code in (404, 405)
