"""Shared fixtures: a throwaway git repo with an `origin` remote and a fake agent CLI."""
import subprocess
import sys
import textwrap
from pathlib import Path
import pytest
from harness.agents import AgentConfig, AgentsConfig
from harness.model import Check, Param, TaskTemplate
from harness.runner import Harness, HarnessSettings


def sh(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


FAKE_AGENT = textwrap.dedent("""
    import sys, pathlib
    workdir, prompt_file, target = sys.argv[1], sys.argv[2], sys.argv[3]
    prompt = pathlib.Path(prompt_file).read_text()
    print("fake agent read", len(prompt), "chars")
    p = pathlib.Path(workdir) / target
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("made by fake agent\\n")
""")


@pytest.fixture
def repo(tmp_path) -> Path:
    origin = tmp_path / "origin.git"
    sh("git", "init", "-q", "--bare", "-b", "main", str(origin), cwd=tmp_path)
    work = tmp_path / "repo"
    sh("git", "init", "-q", "-b", "main", str(work), cwd=tmp_path)
    for k, v in (("user.name", "Tester"), ("user.email", "t@example.com")):
        sh("git", "config", k, v, cwd=work)
    (work / "README.md").write_text("hello\n")
    sh("git", "add", "-A", cwd=work)
    sh("git", "commit", "-q", "-m", "init", cwd=work)
    sh("git", "remote", "add", "origin", str(origin), cwd=work)
    sh("git", "push", "-q", "-u", "origin", "main", cwd=work)
    return work


def make_harness(tmp_path: Path, repo: Path, target: str = "configs/new.txt",
                 checks: list[Check] | None = None) -> Harness:
    agent_py = tmp_path / "fake_agent.py"
    agent_py.write_text(FAKE_AGENT)
    agents = AgentsConfig(default="fake", agents={
        "fake": AgentConfig(name="fake", argv=[sys.executable, str(agent_py), "{workdir}", "{prompt_file}", target])})
    template = TaskTemplate(
        id="demo", title="Demo task", description="write a file",
        params=[Param(name="name", type="str"), Param(name="n", type="int", default=2, required=False)],
        context=["README.md"], prompt="Create {name} x{n}.", allowed_paths=["configs/"],
        acceptance=checks if checks is not None else [Check(name="exists", cmd="test -f configs/new.txt")])
    settings = HarnessSettings(repo=repo, var_dir=tmp_path / "var", worktrees_dir=tmp_path / "wts")
    return Harness(settings, {"demo": template}, agents)
