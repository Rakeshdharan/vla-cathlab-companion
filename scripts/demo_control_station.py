"""Try the Control Station UI safely: a throwaway git repo and a fake agent (no Copilot login, no real repo changes).

    uv run python scripts/demo_control_station.py      ->  http://localhost:8700

Pick AGENT -> "Free task", set paths `configs/` and check `test -f configs/new.txt`, then Start agent.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn  # noqa: E402
from harness.agents import AgentConfig, AgentsConfig  # noqa: E402
from harness.model import load_templates  # noqa: E402
from harness.runner import Harness, HarnessSettings  # noqa: E402
from orchestrator.control import create_control_app  # noqa: E402

FAKE_AGENT = (
    "import sys, pathlib, time\n"
    "print('fake agent working...'); time.sleep(2)\n"
    "p = pathlib.Path(sys.argv[1]) / 'configs/new.txt'\n"
    "p.parent.mkdir(parents=True, exist_ok=True)\n"
    "p.write_text('hello from fake agent\\n'); print('done')\n"
)


def sh(*args: str, cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="cathlab-demo-"))
    sh("git", "init", "-q", "--bare", "-b", "main", str(tmp / "origin.git"), cwd=tmp)
    repo = tmp / "repo"
    sh("git", "init", "-q", "-b", "main", str(repo), cwd=tmp)
    sh("git", "config", "user.name", "Demo", cwd=repo)
    sh("git", "config", "user.email", "demo@example.com", cwd=repo)
    (repo / "README.md").write_text("demo\n")
    sh("git", "add", "-A", cwd=repo)
    sh("git", "commit", "-q", "-m", "init", cwd=repo)
    sh("git", "remote", "add", "origin", str(tmp / "origin.git"), cwd=repo)
    sh("git", "push", "-q", "-u", "origin", "main", cwd=repo)
    agent_py = tmp / "fake_agent.py"
    agent_py.write_text(FAKE_AGENT)
    agents = AgentsConfig(default="fake", agents={"fake": AgentConfig(
        name="fake", description="demo agent (writes configs/new.txt)", argv=[sys.executable, str(agent_py), "{workdir}"])})
    harness = Harness(HarnessSettings(repo=repo, var_dir=tmp / "var", worktrees_dir=tmp / "wts"),
                      load_templates(ROOT / "harness/tasks"), agents)
    print(f"demo repo: {repo}\nopen http://localhost:8700")
    uvicorn.run(create_control_app(ROOT, var_dir=tmp / "control", harness=harness), host="127.0.0.1", port=8700,
                log_level="warning")


if __name__ == "__main__":
    main()
