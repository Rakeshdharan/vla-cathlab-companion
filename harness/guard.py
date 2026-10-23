"""Path guard: every changed file must match an allowed pattern.

Pattern forms: 'dir/' = anything under dir; otherwise fnmatch ('*' also matches '/').

Absolute paths, Windows drive/backslash paths, and any path containing a '..'
component are always rejected, regardless of the allowed patterns."""
import re
from fnmatch import fnmatch


def _is_unsafe(path: str) -> bool:
    if path.startswith("/") or "\\" in path:
        return True
    if re.search(r"(?:^|/)\.\.(?:/|$)", path):
        return True
    if re.match(r"^[A-Za-z]:", path):
        return True
    return False


def is_allowed(path: str, allowed: list[str]) -> bool:
    if _is_unsafe(path):
        return False
    for pattern in allowed:
        if pattern.endswith("/") and path.startswith(pattern):
            return True
        if fnmatch(path, pattern):
            return True
    return False


def path_violations(files: list[str], allowed: list[str]) -> list[str]:
    return [f for f in files if not is_allowed(f, allowed)]
