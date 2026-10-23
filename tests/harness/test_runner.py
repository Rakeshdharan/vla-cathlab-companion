import subprocess
import os
import sys
import textwrap
import threading
import time
from pathlib import Path
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


# Fix 1: Secret leakage - env filtering
def test_env_allow_filters_secrets_but_keeps_safe_vars(tmp_path, repo):
    """Test that os.environ leakage is prevented; only allow-listed vars are passed to agent."""
    # Save original env
    old_token = os.environ.get("HF_TOKEN")
    old_path = os.environ.get("PATH")
    
    try:
        os.environ["HF_TOKEN"] = "leak-this-should-not-appear"
        os.environ["PATH"] = old_path or "/bin:/usr/bin"
        
        # Fake agent that dumps HF_TOKEN to a file
        agent_py = tmp_path / "dump_env.py"
        agent_py.write_text(textwrap.dedent("""
            import sys, pathlib, os
            workdir, prompt_file, target = sys.argv[1], sys.argv[2], sys.argv[3]
            p = pathlib.Path(workdir) / target
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"HF_TOKEN={os.environ.get('HF_TOKEN')},PATH={'PATH' in os.environ}\\n")
        """))
        
        from harness.agents import AgentConfig, AgentsConfig
        agents = AgentsConfig(default="dump", agents={
            "dump": AgentConfig(name="dump", argv=[sys.executable, str(agent_py), "{workdir}", "{prompt_file}", "configs/new.txt"])
        })
        
        from harness.model import TaskTemplate, Param
        template = TaskTemplate(
            id="demo", title="Demo", description="test",
            params=[Param(name="x", type="str")],
            context=[], prompt="test", allowed_paths=["configs/"],
            acceptance=[Check(name="check", cmd="test -f configs/new.txt")]
        )
        
        from harness.runner import HarnessSettings, Harness
        settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
        h = Harness(settings, {"demo": template}, agents)
        
        job = h.run(h.create("demo", {"x": "y"}).id)
        assert job.state == "ready_for_review", job.error
        
        # Check that HF_TOKEN was NOT passed (should be None) but PATH is present
        content = (Path(job.worktree) / "configs/new.txt").read_text()
        assert "HF_TOKEN=None" in content
        assert "PATH=True" in content
    finally:
        if old_token is not None:
            os.environ["HF_TOKEN"] = old_token
        elif "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]


# Fix 2: Agent self-commit detection
def test_agent_self_commit_fails_job(tmp_path, repo):
    """Test that if agent creates commits, the job fails."""
    agent_py = tmp_path / "committer.py"
    agent_py.write_text(textwrap.dedent("""
        import sys, pathlib, subprocess
        workdir, prompt_file, target = sys.argv[1], sys.argv[2], sys.argv[3]
        p = pathlib.Path(workdir) / target
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("made by agent\\n")
        # Agent commits itself (it should not!)
        subprocess.run(["git", "add", "-A"], cwd=workdir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "agent commit"], cwd=workdir, check=True, capture_output=True)
    """))
    
    from harness.agents import AgentConfig, AgentsConfig
    agents = AgentsConfig(default="bad", agents={
        "bad": AgentConfig(name="bad", argv=[sys.executable, str(agent_py), "{workdir}", "{prompt_file}", "configs/new.txt"])
    })
    
    from harness.model import TaskTemplate, Param
    template = TaskTemplate(
        id="demo", title="Demo", description="test",
        params=[Param(name="x", type="str")],
        context=[], prompt="test", allowed_paths=["configs/"],
        acceptance=[Check(name="check", cmd="test -f configs/new.txt")]
    )
    
    from harness.runner import HarnessSettings, Harness
    settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
    h = Harness(settings, {"demo": template}, agents)
    
    job = h.run(h.create("demo", {"x": "y"}).id)
    assert job.state == "failed" and "agent created commits" in job.error


# Fix 2: Push URL guard
def test_worktree_push_disabled_but_harness_push_works(tmp_path, repo):
    """Test that the worktree has push disabled, but Harness.push uses explicit URL."""
    from pathlib import Path
    from harness.model import TaskTemplate, Param
    from harness.agents import AgentConfig, AgentsConfig
    from harness.runner import HarnessSettings, Harness
    from tests.harness.conftest import make_harness
    
    h = make_harness(tmp_path, repo)
    job = h.run(h.create("demo", {"name": "test"}).id)
    assert job.state == "ready_for_review"
    
    wt = Path(job.worktree)
    # Verify that direct push in worktree fails (pushurl is disabled)
    result = subprocess.run(["git", "push", "origin", job.branch], cwd=wt, capture_output=True, text=True)
    assert result.returncode != 0 and "DISABLED" in result.stderr
    
    # But Harness.push should work
    pushed = h.push(job.id)
    assert pushed.state == "pushed"
    
    # Verify main repo's push URL is unchanged
    main_pushurl = subprocess.run(["git", "config", "--get", "remote.origin.pushurl"], cwd=repo,
                                   capture_output=True, text=True).stdout.strip()
    assert main_pushurl == ""


# Fix 3: Force discard of stuck jobs
def test_force_discard_stuck_job_without_thread(tmp_path, repo):
    """Test that force=True allows discarding jobs in intermediate states if no thread is alive."""
    from harness.model import TaskTemplate, Param
    from harness.agents import AgentConfig, AgentsConfig
    from harness.runner import HarnessSettings, Harness
    
    agents = AgentsConfig(default="fake", agents={
        "fake": AgentConfig(name="fake", argv=["true"])
    })
    template = TaskTemplate(
        id="demo", title="Demo", description="test",
        params=[Param(name="x", type="str")],
        context=[], prompt="test", allowed_paths=["configs/"],
        acceptance=[]
    )
    settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
    h = Harness(settings, {"demo": template}, agents)
    
    # Artificially create a job in "agent_running" state
    job = h.create("demo", {"x": "y"})
    job.state = "agent_running"
    h._save(job)
    
    # Discard without force should fail
    with pytest.raises(HarnessError):
        h.discard(job.id)
    
    # Discard with force should work (no thread is alive for this job)
    out = h.discard(job.id, force=True)
    assert out.state == "discarded"


# Fix 4: Atomic job saves with locking
def test_concurrent_saves_produce_valid_json(tmp_path, repo):
    """Test that concurrent _save calls produce valid JSON via locking and atomic writes."""
    from harness.model import TaskTemplate, Param
    from harness.agents import AgentConfig, AgentsConfig
    from harness.runner import HarnessSettings, Harness
    
    agents = AgentsConfig(default="fake", agents={
        "fake": AgentConfig(name="fake", argv=["true"])
    })
    template = TaskTemplate(
        id="demo", title="Demo", description="test",
        params=[Param(name="x", type="str")],
        context=[], prompt="test", allowed_paths=["configs/"],
        acceptance=[]
    )
    settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
    h = Harness(settings, {"demo": template}, agents)
    
    job = h.create("demo", {"x": "y"})
    
    # Launch 20 threads that all save the same job
    def save_job():
        for _ in range(5):
            job.extra = f"update {time.time()}"
            h._save(job)
            time.sleep(0.001)
    
    threads = [threading.Thread(target=save_job) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify the final job.json is valid JSON
    final = h.get(job.id)
    assert final.id == job.id

