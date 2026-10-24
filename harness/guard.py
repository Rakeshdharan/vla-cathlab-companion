"""Path guard: every changed file must match an allowed pattern.

Pattern forms: 'dir/' = anything under dir; otherwise fnmatch ('*' also matches '/').

Absolute paths, Windows drive/backslash paths, any path containing a '..' component, and the FROZEN_PATHS are
always rejected, regardless of the allowed patterns."""
import re
from fnmatch import fnmatch


# Published datasets and eval numbers depend on these; changing them needs a new versioned file instead.
FROZEN_PATHS = ("configs/variation/cathlab_v1.yaml",)


def frozen_violations(files: list[str]) -> list[str]:
    return [f for f in files if f in FROZEN_PATHS]


def _is_unsafe(path: str) -> bool:
    if path.startswith("/") or "\\" in path:
        return True
    if re.search(r"(?:^|/)\.\.(?:/|$)", path):
        return True
    if re.match(r"^[A-Za-z]:", path):
        return True
    return False


def is_allowed(path: str, allowed: list[str]) -> bool:
    if _is_unsafe(path) or path in FROZEN_PATHS:
        return False
    for pattern in allowed:
        if pattern.endswith("/") and path.startswith(pattern):
            return True
        if fnmatch(path, pattern):
            return True
    return False


def path_violations(files: list[str], allowed: list[str]) -> list[str]:
    return [f for f in files if not is_allowed(f, allowed)]
