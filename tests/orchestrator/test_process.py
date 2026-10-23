import sys
from orchestrator.process import read_log, start_process

PY = sys.executable


def test_success_writes_log_and_calls_back(tmp_path):
    seen = []
    h = start_process([PY, "-c", "print('hello'); import sys; print('err', file=sys.stderr)"],
                      tmp_path / "a.log", on_exit=lambda rc, st: seen.append((rc, st)))
    assert h.wait(10) == "done" and h.returncode == 0
    text, off = read_log(tmp_path / "a.log")
    assert "hello" in text and "err" in text and off == len(text.encode())
    assert seen == [(0, "done")]


def test_failure_state(tmp_path):
    h = start_process([PY, "-c", "raise SystemExit(3)"], tmp_path / "b.log")
    assert h.wait(10) == "failed" and h.returncode == 3


def test_shell_string_cwd_and_env(tmp_path):
    h = start_process("echo $GREETING > out.txt", tmp_path / "c.log", cwd=tmp_path, env={"GREETING": "hi"})
    assert h.wait(10) == "done"
    assert (tmp_path / "out.txt").read_text().strip() == "hi"


def test_cancel_and_timeout(tmp_path):
    h = start_process([PY, "-c", "import time; time.sleep(30)"], tmp_path / "d.log")
    h.cancel()
    assert h.wait(10) == "cancelled"
    t = start_process([PY, "-c", "import time; time.sleep(30)"], tmp_path / "e.log", timeout_s=0.3)
    assert t.wait(10) == "timeout"


def test_read_log_offsets_and_append(tmp_path):
    p = tmp_path / "f.log"
    start_process([PY, "-c", "print('one')"], p).wait(10)
    first, off = read_log(p)
    start_process([PY, "-c", "print('two')"], p, append=True).wait(10)
    second, off2 = read_log(p, off)
    assert first.strip() == "one" and second.strip() == "two" and off2 > off
    assert read_log(tmp_path / "missing.log", 5) == ("", 5)
