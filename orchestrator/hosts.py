"""Host-header allow-list for the Control Station servers (blocks DNS-rebinding attacks on localhost).

A rebinding page reaches 127.0.0.1 but still sends its own hostname in `Host`, so rejecting unknown hosts closes it.
Extra names (LAN hostname, IP of a GPU box): CATHLAB_ALLOWED_HOSTS=gpu-box,10.0.0.5
"""
import os
from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

# "testserver" is Starlette's TestClient host; a single-label name an attacker cannot register publicly.
LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]", "testserver"]
WILDCARD_BINDS = {"0.0.0.0", "::", ""}


def allowed_hosts(bind_host: str | None = None) -> list[str]:
    hosts = list(LOCAL_HOSTS)
    if bind_host and bind_host not in WILDCARD_BINDS and bind_host not in hosts:
        hosts.append(bind_host)
    extra = os.environ.get("CATHLAB_ALLOWED_HOSTS", "")
    hosts += [h.strip() for h in extra.split(",") if h.strip() and h.strip() not in hosts]
    return hosts


def protect(app: FastAPI, bind_host: str | None = None) -> FastAPI:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts(bind_host))
    return app
