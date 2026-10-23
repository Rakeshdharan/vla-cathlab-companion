import subprocess
import pytest
from harness.model import Check
from harness.runner import HarnessError
from tests.harness.conftest import make_harness


def test_happy_path_ready_for_review_then_push(tmp_path, repo):
    h = make_harness(tmp_path, repo)
    job = h.create("demo", {"name": "hoop"}, extra="be quick")
    assert job.branch == f"agent/demo-{job.id}" and job.params == {"name": "hoop", "n": 2}
    prompt = (h.job_dir(job.id) / "prompt.md").read_text()
    assert "Create hoop x2." in prompt and "be quick" in prompt

    done = h.run(job.id)
    assert done.state == "ready_for_review", done.error
    assert done.changed_files == ["configs/new.txt"]
    assert [c.ok for c in done.checks] == [True]
    assert "made by fake agent" in h.diff(job.id)
    text, _ = h.log(job.id)
    assert "fake agent read" in text and "ready for review" in text
    log = subprocess.run(["git", "log", "-1", "--format=%an|%s", done.branch], cwd=repo,
                         capture_output=True, text=True).stdout.strip()
    assert log == f"Tester|agent(demo): Demo task [{job.id}]"

    pushed = h.push(job.id)
    assert pushed.state == "pushed"
    remote = subprocess.run(["git", "ls-remote", "--heads", "origin", done.branch], cwd=repo,
                            capture_output=True, text=True).stdout
    assert done.branch in remote
    assert [j.id for j in h.list_jobs()] == [job.id]


def test_guard_blocks_changes_outside_allowed_paths(tmp_path, repo):
    h = make_harness(tmp_path, repo, target="contracts/hack.py")
    job = h.run(h.create("demo", {"name": "x"}).id)
    assert job.state == "failed" and "outside allowed paths" in job.error and "contracts/hack.py" in job.error


def test_failing_check_fails_job_and_discard_cleans_up(tmp_path, repo):
    h = make_harness(tmp_path, repo, checks=[Check(name="never", cmd="exit 1")])
    job = h.run(h.create("demo", {"name": "x"}).id)
    assert job.state == "failed" and "never" in job.error
    with pytest.raises(HarnessError):
        h.push(job.id)
    out = h.discard(job.id)
    assert out.state == "discarded"
    branches = subprocess.run(["git", "branch", "--list", job.branch], cwd=repo, capture_output=True, text=True).stdout
    assert branches.strip() == "" and not (tmp_path / "wts" / job.id).exists()


def test_unknown_task_and_agent(tmp_path, repo):
    h = make_harness(tmp_path, repo)
    with pytest.raises(HarnessError):
        h.create("nope", {})
    with pytest.raises(KeyError):
        h.create("demo", {"name": "x"}, agent="missing")
