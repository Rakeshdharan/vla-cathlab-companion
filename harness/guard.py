"""Path guard: every changed file must match an allowed pattern.

Pattern forms: 'dir/' = anything under dir; otherwise fnmatch ('*' also matches '/')."""
from fnmatch import fnmatch


def is_allowed(path: str, allowed: list[str]) -> bool:
    for pattern in allowed:
        if pattern.endswith("/") and path.startswith(pattern):
            return True
        if fnmatch(path, pattern):
            return True
    return False


def path_violations(files: list[str], allowed: list[str]) -> list[str]:
    return [f for f in files if not is_allowed(f, allowed)]
