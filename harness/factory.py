"""Build the default Harness for a repo checkout."""
import os
from pathlib import Path
from harness.agents import load_agents
from harness.model import load_templates
from harness.runner import Harness, HarnessSettings

PACKAGE_DIR = Path(__file__).resolve().parent


def default_harness(repo_root: str | Path, var_dir: str | Path | None = None, fetch: bool = True) -> Harness:
    repo_root = Path(repo_root).resolve()
    var = Path(var_dir) if var_dir else repo_root / "var"
    worktrees = Path(os.environ.get("CATHLAB_WORKTREES", repo_root.parent / f".{repo_root.name}-agent-worktrees"))
    settings = HarnessSettings(repo=repo_root, var_dir=var, worktrees_dir=worktrees,
                               base_ref=os.environ.get("CATHLAB_AGENT_BASE", "origin/main"), fetch=fetch)
    return Harness(settings, load_templates(PACKAGE_DIR / "tasks"), load_agents(repo_root / "configs/agents.yaml"))
