"""Policy server: one model version over HTTP + JSON (run it in the lerobot env, next to the GPU).

    python -m serve.policy_server --checkpoint Rakeshdharan56/smolvla_panda_pick_v1 --model-version cathlab-pick@v3
    python -m serve.policy_server --registry cathlab-pick@production          # checkpoint + version from the registry
    # engines then evaluate / roll out with --policy url:http://<host>:8720

GET /health (no token), GET /meta, POST /act {instruction, obs: {key: Array}} -> {actions: Array, model_version,
latency_ms}. Arrays use serve/codec.py. 503 until the model is loaded (it loads in a background thread).
"""
import argparse
import importlib
import os
import secrets
import threading
import time
from pathlib import Path
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel
from orchestrator.hosts import protect
from serve.codec import decode_array, encode_array

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = "train.lerobot_adapter:LeRobotPolicyAdapter"


class ActBody(BaseModel):
    instruction: str
    obs: dict[str, dict]


def load_adapter_class(spec: str) -> type:
    module, sep, cls = spec.partition(":")
    if not sep:
        module, _, cls = spec.rpartition(".")
    return getattr(importlib.import_module(module), cls)


def create_policy_app(adapter_spec: str, checkpoint: str, model_version: str, io: str = "panda_libero",
                      token: str | None = None, bind_host: str | None = None) -> FastAPI:
    app = FastAPI(title=f"cathlab policy server: {model_version}")
    if not token:
        protect(app, bind_host)
    state = {"ready": False, "error": "", "adapter": None}

    def _load() -> None:
        try:
            adapter = load_adapter_class(adapter_spec)()
            adapter.load(checkpoint)
            state.update(adapter=adapter, ready=True)
        except Exception as e:  # noqa: BLE001 - reported on /health
            state["error"] = f"{type(e).__name__}: {e}"[:500]

    threading.Thread(target=_load, daemon=True).start()

    def auth(x_cathlab_token: str | None = Header(None)) -> None:
        if token and (x_cathlab_token is None or not secrets.compare_digest(x_cathlab_token, token)):
            raise HTTPException(401, "missing or invalid X-Cathlab-Token")

    def adapter_or_503():
        if not state["ready"]:
            raise HTTPException(503, state["error"] or "model loading")
        return state["adapter"]

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "ready": state["ready"], "model_version": model_version, "auth": bool(token),
                "error": state["error"]}

    @app.get("/meta")
    def meta(_: None = Depends(auth)) -> dict:
        adapter = adapter_or_503()
        return {"model_version": model_version, "io": io, "adapter": adapter_spec, "checkpoint": checkpoint,
                "meta": adapter.meta().model_dump()}

    @app.post("/act")
    def act(body: ActBody, _: None = Depends(auth)) -> dict:
        adapter = adapter_or_503()
        try:
            obs = {k: decode_array(v) for k, v in body.obs.items()}
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(400, f"bad obs: {e}") from e
        t0 = time.perf_counter()
        try:
            actions = np.asarray(adapter.predict(obs, body.instruction), dtype=np.float32)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"predict failed: {type(e).__name__}: {e}"[:500]) from e
        if actions.ndim == 1:
            actions = actions[None, :]
        return {"actions": encode_array(actions), "model_version": model_version,
                "latency_ms": (time.perf_counter() - t0) * 1000}

    return app


def main(argv: list[str] | None = None) -> None:
    import uvicorn
    import yaml
    from orchestrator.worker import check_bind

    cfg = yaml.safe_load((ROOT / "configs/serve.yaml").read_text()) or {}
    ap = argparse.ArgumentParser(prog="python -m serve.policy_server", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER, help="module:Class with no-arg constructor + load(ckpt)")
    ap.add_argument("--checkpoint", help="HF repo id or local path")
    ap.add_argument("--model-version", help="label returned to clients, e.g. cathlab-pick@v3")
    ap.add_argument("--registry", help="serve a registry version instead, e.g. cathlab-pick@production")
    ap.add_argument("--registry-root", default=str(ROOT / cfg.get("registry_path", "var/registry")))
    ap.add_argument("--io", default="panda_libero")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(cfg.get("policy_server_port", 8720)))
    ap.add_argument("--token", default=None, help="defaults to CATHLAB_WORKER_TOKEN")
    a = ap.parse_args(argv)
    if a.registry:
        from serve.registry import Registry
        m = Registry(a.registry_root).resolve(a.registry)
        if m.card.adapter == "expert":
            raise SystemExit("the scripted expert needs simulator state; it can't be served")
        a.checkpoint, a.model_version = m.card.ckpt_uri, f"{m.card.model_name}@{m.card.version}"
    if not a.checkpoint:
        raise SystemExit("--checkpoint or --registry is required")
    token = a.token or os.environ.get("CATHLAB_WORKER_TOKEN")
    check_bind(a.host, token)
    app = create_policy_app(a.adapter, a.checkpoint, a.model_version or a.checkpoint, a.io, token, a.host)
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
