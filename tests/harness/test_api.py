import time
from fastapi.testclient import TestClient
from harness.server import create_agent_app
from harness.__main__ import main
from tests.harness.conftest import make_harness


def wait(c, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/api/agent/jobs/{job_id}").json()["state"]
        if st in ("ready_for_review", "failed"):
            return st
        time.sleep(0.05)
    return st


def test_agent_api_end_to_end(tmp_path, repo):
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    assert c.get("/health").json()["capabilities"] == ["harness"]
    assert [t["id"] for t in c.get("/api/agent/tasks").json()] == ["demo"]
    assert c.get("/api/agent/agents").json()["default"] == "fake"

    bad = c.post("/api/agent/jobs", json={"task": "demo", "params": {}})
    assert bad.status_code == 400 and "missing required" in bad.json()["detail"]

    job = c.post("/api/agent/jobs", json={"task": "demo", "params": {"name": "hoop"}}).json()
    assert wait(c, job["id"]) == "ready_for_review"
    assert "Create hoop" in c.get(f"/api/agent/jobs/{job['id']}/prompt").text
    assert "made by fake agent" in c.get(f"/api/agent/jobs/{job['id']}/diff").text
    log = c.get(f"/api/agent/jobs/{job['id']}/log").json()
    assert "ready for review" in log["text"] and log["offset"] > 0
    assert c.post(f"/api/agent/jobs/{job['id']}/push").json()["state"] == "pushed"
    assert c.post(f"/api/agent/jobs/{job['id']}/push").status_code == 409
    assert c.get("/api/agent/jobs/unknown").status_code == 404


def test_api_discard_force(tmp_path, repo):
    """Test that discard with force=true succeeds even when agent_running, discard without force fails."""
    harness = make_harness(tmp_path, repo)
    c = TestClient(create_agent_app(harness))

    # Create and start a job
    job = c.post("/api/agent/jobs", json={"task": "demo", "params": {"name": "test"}}).json()
    job_id = job["id"]

    # Manually set job state to "agent_running" and remove thread to simulate stuck state
    job_obj = harness.get(job_id)
    job_obj.state = "agent_running"
    with harness._lock:
        # Remove thread if any to simulate the stuck state
        if job_id in harness._threads:
            del harness._threads[job_id]

    # Try to discard without force - should fail with 409
    resp = c.post(f"/api/agent/jobs/{job_id}/discard")
    assert resp.status_code == 409
    assert "still" in resp.json()["detail"]

    # Discard with force=true - should succeed with 200
    resp = c.post(f"/api/agent/jobs/{job_id}/discard?force=true")
    assert resp.status_code == 200
    assert resp.json()["state"] == "discarded"


def test_cli_discard_force_flag(tmp_path, repo, monkeypatch):
    """Test that the real CLI discard command with --force flag works end-to-end."""
    import time
    from pathlib import Path
    from harness.factory import default_harness
    from harness.runner import AgentJob, HarnessError
    import pytest

    # Monkeypatch away CATHLAB_WORKTREES if set
    monkeypatch.delenv("CATHLAB_WORKTREES", raising=False)

    # Copy real agents.yaml into the test repo
    agents_src = Path(__file__).parent.parent.parent / "configs/agents.yaml"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    (repo / "configs/agents.yaml").write_text(agents_src.read_text())

    # Build the harness that the CLI will build
    h = default_harness(repo, fetch=False)

    # Create a stuck job by manually saving an AgentJob record
    job_id = "test-stuck-job-1234"
    now = time.time()
    stuck_job = AgentJob(
        id=job_id,
        task="demo",
        title="Stuck job",
        params={"name": "test"},
        extra="",
        agent="copilot",
        branch=f"agent/demo-{job_id}",
        worktree=str(tmp_path / "wt" / job_id),
        state="agent_running",
        created=now,
        updated=now,
    )
    h._save(stuck_job)

    # Try to discard without force - should fail with HarnessError
    with pytest.raises(HarnessError):
        main(["--repo", str(repo), "--no-fetch", "discard", job_id])

    # Discard with force - should succeed with exit code 0
    rc = main(["--repo", str(repo), "--no-fetch", "discard", job_id, "--force"])
    assert rc == 0

    # Verify the job state is now "discarded"
    h2 = default_harness(repo, fetch=False)
    job = h2.get(job_id)
    assert job.state == "discarded"




def test_push_git_failure_returns_502_with_reason(tmp_path, repo):
    import subprocess
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    job = c.post("/api/agent/jobs", json={"task": "demo", "params": {"name": "hoop"}}).json()
    assert wait(c, job["id"]) == "ready_for_review"
    subprocess.run(["git", "config", "remote.origin.url", str(tmp_path / "missing.git")], cwd=repo, check=True)
    r = c.post(f"/api/agent/jobs/{job['id']}/push")
    assert r.status_code == 502 and "git push" in r.json()["detail"]


def test_git_errors_redact_url_credentials(tmp_path):
    import pytest
    from harness.gitops import GitError, git
    with pytest.raises(GitError) as e:
        git(["ls-remote", "https://user:s3cret@127.0.0.1:9/x.git"], tmp_path)
    assert "s3cret" not in str(e.value) and "://***@" in str(e.value)


def test_agent_app_rejects_foreign_host_header(tmp_path, repo):
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    assert c.get("/health").status_code == 200
    assert c.get("/health", headers={"Host": "attacker.example"}).status_code == 400
