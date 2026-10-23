import socket
import threading
import time
import numpy as np
import pytest
import uvicorn
from contracts import DummySim, TaskSpec, load_yaml
from engines.policies import make_policy
from serve.codec import decode_array, encode_array
from serve.policy_client import PolicyClient
from serve.policy_server import create_policy_app


class SlowRandom:
    """Adapter whose load takes 0.6 s, to exercise 503 + client retry."""
    def __init__(self):
        from contracts import RandomPolicy
        self.inner = RandomPolicy(action_dim=7, chunk_size=4)

    def load(self, ckpt):
        time.sleep(0.6)

    def predict(self, obs, instruction):
        assert obs["state"].shape == (8,) and obs["agentview"].dtype == np.uint8
        return self.inner.predict(obs, instruction)

    def meta(self):
        return self.inner.meta()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    started = []

    def start(adapter, token=None):
        port = free_port()
        app = create_policy_app(adapter, "ckpt", "cathlab-pick@v9", token=token, bind_host="127.0.0.1")
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        threading.Thread(target=srv.run, daemon=True).start()
        end = time.time() + 10
        while not srv.started and time.time() < end:
            time.sleep(0.02)
        started.append(srv)
        return f"http://127.0.0.1:{port}"

    yield start
    for srv in started:
        srv.should_exit = True


def test_codec_round_trip():
    for a in (np.arange(6, dtype=np.float32).reshape(2, 3), np.zeros((4, 4, 3), np.uint8), np.array([1, 2], np.int64)):
        b = decode_array(encode_array(a))
        assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b) and b.flags.c_contiguous
    with pytest.raises(ValueError):
        encode_array(np.zeros(2, dtype=np.complex64))
    with pytest.raises(ValueError):
        decode_array({**encode_array(np.zeros(3, np.float32)), "shape": [4]})


def test_client_waits_for_load_and_predicts(server):
    url = server(f"{__name__}:SlowRandom")
    client = PolicyClient(url, wait_ready_s=10)
    obs = {"state": np.zeros(8, np.float32), "agentview": np.zeros((8, 8, 3), np.uint8)}
    actions = client.predict(obs, "Hand me the vial")                        # retries 503 while loading
    assert actions.shape == (4, 7) and actions.dtype == np.float32
    assert client.meta().action_dim == 7
    assert make_policy(f"url:{url}", action_dim=7).predict(obs, "x").shape == (4, 7)


def test_token_and_errors(server):
    url = server("contracts.policy:RandomPolicy", token="s3cret")
    with pytest.raises(RuntimeError, match="401"):
        PolicyClient(url, token="wrong", wait_ready_s=2).predict({"state": np.zeros(8, np.float32)}, "x")
    client = PolicyClient(url, token="s3cret", wait_ready_s=5)
    client.load()
    assert client.predict({"state": np.zeros(8, np.float32)}, "x").shape == (10, 16)
    with pytest.raises(RuntimeError, match="not ready"):
        PolicyClient("http://127.0.0.1:9", wait_ready_s=0.3).load()


def test_eval_runner_works_through_the_client(server):
    from eval.runner import run_suite
    url = server("contracts.policy:RandomPolicy")
    task = load_yaml("configs/tasks/cathlab_pick.yaml", TaskSpec)
    scene = load_yaml("configs/scenes/dummy.yaml", __import__("contracts").SceneSpec)
    report = run_suite(PolicyClient(url, wait_ready_s=5), DummySim(), scene, task, [0], "cathlab-pick@v9", [])
    assert len(report.episodes) == len(task.objects)
