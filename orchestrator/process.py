"""Run one subprocess with its output in a log file; used by workers and the agent harness."""
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Callable


class ProcessHandle:
    def __init__(self, proc: subprocess.Popen, log_path: Path, timeout_s: float | None,
                 on_exit: Callable[[int, str], None] | None, kill_grace_s: float = 5.0):
        self.proc, self.log_path = proc, log_path
        self.state = "running"  # running -> done | failed | cancelled | timeout
        self.returncode: int | None = None
        self._cancelled = False
        self._timed_out = False
        self._on_exit = on_exit
        self._done = threading.Event()
        self._kill_grace_s = kill_grace_s
        self._kill_timer = None
        self._timer = threading.Timer(timeout_s, self._timeout) if timeout_s else None
        if self._timer:
            self._timer.daemon = True
            self._timer.start()
        threading.Thread(target=self._watch, daemon=True).start()

    @property
    def pid(self) -> int:
        return self.proc.pid

    def _send_sig(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def _escalate_to_kill(self) -> None:
        if self.proc.poll() is None:
            self._send_sig(signal.SIGKILL)

    def _timeout(self) -> None:
        self._timed_out = True
        if self.proc.poll() is None:
            self._send_sig(signal.SIGTERM)
            self._kill_timer = threading.Timer(self._kill_grace_s, self._escalate_to_kill)
            self._kill_timer.daemon = True
            self._kill_timer.start()

    def cancel(self) -> None:
        if self.proc.poll() is None:
            self._cancelled = True
            self._send_sig(signal.SIGTERM)
            self._kill_timer = threading.Timer(self._kill_grace_s, self._escalate_to_kill)
            self._kill_timer.daemon = True
            self._kill_timer.start()

    def _watch(self) -> None:
        rc = self.proc.wait()
        if self._timer:
            self._timer.cancel()
        if self._kill_timer:
            self._kill_timer.cancel()
        self.returncode = rc
        
        if rc == 0:
            self.state = "done"
        elif self._cancelled:
            self.state = "cancelled"
        elif self._timed_out:
            self.state = "timeout"
        else:
            self.state = "failed"
        
        if self._on_exit:
            self._on_exit(rc, self.state)
        self._done.set()

    def wait(self, timeout: float | None = None) -> str:
        self._done.wait(timeout)
        return self.state


def start_process(cmd: list[str] | str, log_path: str | Path, cwd: str | Path | None = None,
                  env: dict[str, str] | None = None, timeout_s: float | None = None,
                  on_exit: Callable[[int, str], None] | None = None, append: bool = False,
                  kill_grace_s: float = 5.0, inherit_env: bool = True) -> ProcessHandle:
    """Start `cmd` (argv list, or a shell string) with stdout+stderr appended to `log_path`."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab" if append else "wb")
    child_env = {**os.environ, **(env or {})} if inherit_env else (env or {})
    proc = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, env=child_env,
                            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    log.close()
    return ProcessHandle(proc, log_path, timeout_s, on_exit, kill_grace_s)


def read_log(path: str | Path, offset: int = 0, max_bytes: int = 256_000) -> tuple[str, int]:
    """Return (new text since `offset`, next offset). Missing file -> ('', offset)."""
    path = Path(path)
    if not path.exists():
        return "", offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(max_bytes)
    return data.decode("utf-8", errors="replace"), offset + len(data)
