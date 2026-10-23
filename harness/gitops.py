"""Git operations for harness jobs (worktree per job, commit, diff, push, cleanup)."""
import subprocess
from pathlib import Path


class GitError(RuntimeError):
    pass


def git(args: list[str], cwd: str | Path) -> str:
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise GitError(f"git {' '.join(args)}: {res.stderr.strip() or res.stdout.strip()}")
    return res.stdout


def create_worktree(repo: Path, path: Path, branch: str, base_ref: str, fetch: bool = True,
                    remote: str = "origin") -> str:
    """Create `path` on new `branch` from `base_ref`; return the base commit sha."""
    if fetch:
        git(["fetch", "-q", remote], repo)
    base_sha = git(["rev-parse", base_ref], repo).strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    git(["worktree", "add", "-q", "-b", branch, str(path), base_sha], repo)
    return base_sha


def stage_all(worktree: Path) -> list[str]:
    """`git add -A` and return the staged paths (repo-relative, posix)."""
    git(["add", "-A"], worktree)
    out = git(["diff", "--cached", "--name-only"], worktree)
    return [line for line in out.splitlines() if line]


def commit(worktree: Path, message: str) -> str:
    git(["commit", "-q", "-m", message], worktree)
    return git(["rev-parse", "HEAD"], worktree).strip()


def diff(worktree: Path, base_sha: str, stat: bool = False) -> str:
    return git(["diff", "--stat" if stat else "--patch", f"{base_sha}..HEAD"], worktree)


def push(worktree: Path, branch: str, remote: str = "origin", url: str | None = None) -> None:
    if url:
        git(["push", "-q", "-u", url, branch], worktree)
    else:
        git(["push", "-q", "-u", remote, branch], worktree)


def remove_worktree(repo: Path, path: Path, branch: str, delete_branch: bool = True) -> None:
    if path.exists():
        git(["worktree", "remove", "--force", str(path)], repo)
    git(["worktree", "prune"], repo)
    if delete_branch:
        try:
            git(["branch", "-D", branch], repo)
        except GitError:
            pass
