"""PolicyClient: a contracts.PolicyAdapter that calls serve/policy_server.py over HTTP (stdlib + numpy only, so it
runs inside every engine env). Engines use it via --policy url:<server>."""
import json
import os
import time
import urllib.error
import urllib.request
import numpy as np
from contracts import PolicyAdapter, PolicyMeta
from serve.codec import decode_array, encode_array


class PolicyClient(PolicyAdapter):
    def __init__(self, url: str, token: str | None = None, timeout_s: float = 30.0, wait_ready_s: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("CATHLAB_WORKER_TOKEN")
        self.timeout_s, self.wait_ready_s = timeout_s, wait_ready_s

    def _request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.token:
            req.add_header("X-Cathlab-Token", self.token)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                payload = json.loads(e.read() or b"{}")
            except ValueError:
                payload = {}
            return e.code, payload

    def load(self, ckpt: str = "") -> None:
        """Wait until the server has loaded its model (the checkpoint is chosen on the server)."""
        deadline, last = time.time() + self.wait_ready_s, "no answer"
        while True:
            try:
                status, health = self._request("GET", "/health")
                if status == 200 and health.get("ready"):
                    return
                last = health.get("error") or "model loading"
            except (urllib.error.URLError, OSError) as e:
                last = str(e)
            if time.time() >= deadline:
                raise RuntimeError(f"policy server at {self.url} not ready: {last}")
            time.sleep(0.25)

    def predict(self, obs: dict, instruction: str) -> np.ndarray:
        body = {"instruction": instruction, "obs": {k: encode_array(np.asarray(v)) for k, v in obs.items()}}
        deadline = time.time() + self.wait_ready_s
        while True:
            status, payload = self._request("POST", "/act", body)
            if status == 200:
                return decode_array(payload["actions"]).astype(np.float32, copy=False)
            if status == 503 and time.time() < deadline:
                time.sleep(0.25)
                continue
            raise RuntimeError(f"policy server {status}: {payload.get('detail', payload)}")

    def meta(self) -> PolicyMeta:
        status, payload = self._request("GET", "/meta")
        if status != 200:
            raise RuntimeError(f"policy server {status}: {payload.get('detail', payload)}")
        return PolicyMeta(**payload["meta"])
