import sys
import time
from fastapi.testclient import TestClient
from orchestrator.worker import create_worker_app

PY = sys.executable


def wait_state(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        st = client.get(f"/jobs/{job_id}").json()["state"]
        if st != "running":
            return st
        time.sleep(0.05)
    return "running"


def test_health_reports_capabilities(tmp_path):
    c = TestClient(create_worker_app("robosuite", ["generate", "eval"], tmp_path))
    h = c.get("/health").json()
    assert h["name"] == "robosuite" and h["capabilities"] == ["generate", "eval"] and h["jobs_running"] == 0


def test_job_runs_and_log_is_readable_incrementally(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path))
    job = c.post("/jobs", json={"argv": [PY, "-c", "print('stage-ok')"], "name": "generate"}).json()
    assert job["name"] == "generate"
    assert wait_state(c, job["id"]) == "done"
    log = c.get(f"/jobs/{job['id']}/log").json()
    assert "stage-ok" in log["text"] and log["state"] == "done"
    assert c.get(f"/jobs/{job['id']}/log", params={"offset": log["offset"]}).json()["text"] == ""


def test_cancel_unknown_and_empty(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path))
    job = c.post("/jobs", json={"argv": [PY, "-c", "import time; time.sleep(30)"]}).json()
    assert c.post(f"/jobs/{job['id']}/cancel").status_code == 200
    assert wait_state(c, job["id"]) == "cancelled"
    assert c.get("/jobs/nope").status_code == 404
    assert c.post("/jobs", json={"argv": []}).status_code == 400
