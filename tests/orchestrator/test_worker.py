import sys
import time
import threading
from fastapi.testclient import TestClient
from orchestrator.worker import create_worker_app, check_bind
import pytest

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


def test_token_auth_required_for_non_health_endpoints(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path, token="s3cret"))
    
    # POST /jobs without token → 401
    r = c.post("/jobs", json={"argv": [PY, "-c", "print('x')"]})
    assert r.status_code == 401
    assert "X-Cathlab-Token" in r.json()["detail"]
    
    # POST /jobs with wrong token → 401
    r = c.post("/jobs", json={"argv": [PY, "-c", "print('x')"]}, headers={"X-Cathlab-Token": "wrong"})
    assert r.status_code == 401
    
    # POST /jobs with correct token → 200
    r = c.post("/jobs", json={"argv": [PY, "-c", "print('x')"]}, headers={"X-Cathlab-Token": "s3cret"})
    assert r.status_code == 200
    
    # GET /health works without token and reports auth true
    h = c.get("/health").json()
    assert h["auth"] is True


def test_health_reports_auth_false_when_no_token(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path, token=None))
    h = c.get("/health").json()
    assert h["auth"] is False


def test_max_jobs_evicts_oldest_non_running(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path, max_jobs=2))
    
    # Submit two quick jobs
    job1 = c.post("/jobs", json={"argv": [PY, "-c", "print('1')"]}).json()
    job2 = c.post("/jobs", json={"argv": [PY, "-c", "print('2')"]}).json()
    
    # Wait for both to complete
    assert wait_state(c, job1["id"]) == "done"
    assert wait_state(c, job2["id"]) == "done"
    
    # Submit third job (should evict job1 since max_jobs=2)
    job3 = c.post("/jobs", json={"argv": [PY, "-c", "print('3')"]}).json()
    assert job3["state"] == "running" or job3["state"] == "done"
    
    # job1 should now be evicted and return 404
    assert c.get(f"/jobs/{job1['id']}").status_code == 404
    
    # job3 should still be accessible
    assert c.get(f"/jobs/{job3['id']}").status_code == 200


def test_too_many_running_jobs_returns_429(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path, max_jobs=1))
    
    # Submit one long-running job
    job1 = c.post("/jobs", json={"argv": [PY, "-c", "import time; time.sleep(30)"]}).json()
    assert job1["state"] == "running"
    
    try:
        # Try to submit another while first is running → 429
        r = c.post("/jobs", json={"argv": [PY, "-c", "print('x')"]})
        assert r.status_code == 429
        assert "too many running jobs" in r.json()["detail"]
    finally:
        # Clean up: cancel the long-running job
        c.post(f"/jobs/{job1['id']}/cancel")


def test_concurrent_job_submissions_with_eviction(tmp_path):
    """Test concurrent submissions don't race during eviction."""
    c = TestClient(create_worker_app("w", [], tmp_path, max_jobs=3))
    
    # Fill with 3 finished jobs
    for _ in range(3):
        job = c.post("/jobs", json={"argv": [PY, "-c", "print('x')"]}).json()
        wait_state(c, job["id"])
    
    # Submit 6 jobs concurrently from threads
    results = []
    
    def submit_job(index):
        try:
            r = c.post("/jobs", json={"argv": [PY, "-c", f"print({index})"]})
            results.append(r.status_code)
        except Exception as e:
            results.append(f"EXC:{repr(e)}")
    
    threads = [threading.Thread(target=submit_job, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify exactly 6 results
    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    
    # Every response must be 200 or 429 (no exceptions, no 500 errors)
    for i, result in enumerate(results):
        assert isinstance(result, int), f"Result {i} is exception: {result}"
        assert result in (200, 429), f"Result {i} has unexpected status {result}"
    
    # Verify final state: jobs table should have at most max_jobs entries
    h = c.get("/health").json()
    assert h["jobs_total"] <= 3, f"Jobs table size {h['jobs_total']} exceeds max_jobs=3"


def test_failed_process_start_returns_500_and_no_slot_leak(tmp_path):
    """Test that failed start_process doesn't leak a slot and returns 500."""
    c = TestClient(create_worker_app("w", [], tmp_path))
    
    # Get initial table size
    h1 = c.get("/health").json()
    initial_size = h1["jobs_total"]
    
    # Try to start non-existent binary → 500
    r = c.post("/jobs", json={"argv": ["/nonexistent/binary", "arg"]})
    assert r.status_code == 500
    assert "failed to start" in r.json()["detail"]
    
    # Verify table size unchanged (no slot leaked)
    h2 = c.get("/health").json()
    assert h2["jobs_total"] == initial_size, f"Table size changed from {initial_size} to {h2['jobs_total']}"


def test_check_bind_allows_local_hosts_without_token():
    check_bind("127.0.0.1", None)
    check_bind("localhost", None)
    check_bind("::1", None)


def test_check_bind_allows_any_host_with_token():
    check_bind("0.0.0.0", "x")
    check_bind("192.168.1.1", "token")


def test_check_bind_rejects_non_local_host_without_token():
    with pytest.raises(SystemExit):
        check_bind("0.0.0.0", None)
    with pytest.raises(SystemExit):
        check_bind("192.168.1.1", None)


def test_tokenless_worker_rejects_foreign_host_header(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path))
    assert c.get("/health", headers={"Host": "localhost:8711"}).status_code == 200
    assert c.get("/health", headers={"Host": "attacker.example"}).status_code == 400
    assert c.post("/jobs", json={"argv": ["true"]}, headers={"Host": "attacker.example"}).status_code == 400


def test_token_worker_accepts_remote_host_names(tmp_path):
    c = TestClient(create_worker_app("w", [], tmp_path, token="s3cret"))
    assert c.get("/health", headers={"Host": "gpu-box:8712"}).status_code == 200


def test_allowed_hosts_adds_bind_host_and_env(monkeypatch):
    from orchestrator.hosts import allowed_hosts
    monkeypatch.setenv("CATHLAB_ALLOWED_HOSTS", "gpu-box, 10.0.0.5")
    hosts = allowed_hosts("192.168.1.7")
    assert {"localhost", "127.0.0.1", "192.168.1.7", "gpu-box", "10.0.0.5"} <= set(hosts)
    assert "0.0.0.0" not in allowed_hosts("0.0.0.0")
