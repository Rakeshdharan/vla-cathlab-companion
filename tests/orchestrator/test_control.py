import threading
import time
from pathlib import Path
import pytest
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
    start = time.perf_counter()
    while not server.started:
        if time.perf_counter() - start > 5.0:
            pytest.fail("worker test server did not start within 5 s")
        time.sleep(0.01)
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


def test_workers_probed_concurrently(tmp_path):
    """Verify 4 offline workers respond in < 1.5s (concurrent, not serial)."""
    workers_yaml = "workers:\n"
    for i in range(1, 5):
        workers_yaml += f"  - {{name: offline{i}, url: 'http://127.0.0.1:{i}'}}\n"
    c = client(tmp_path, workers_yaml)
    start = time.perf_counter()
    ws = c.get("/api/workers").json()
    elapsed = time.perf_counter() - start
    assert len(ws) == 4
    assert all(not w["healthy"] for w in ws)
    assert [w["name"] for w in ws] == ["offline1", "offline2", "offline3", "offline4"]
    assert elapsed < 1.5, f"Worker probes took {elapsed:.2f}s, expected < 1.5s (serial would take ~2s)"



def test_control_rejects_foreign_host_header(tmp_path):
    c = client(tmp_path)
    assert c.get("/api/health", headers={"Host": "localhost:8700"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "attacker.example:8700"}).status_code == 400
    assert c.post("/api/runs", json={"selection": {}}, headers={"Host": "attacker.example"}).status_code == 400
