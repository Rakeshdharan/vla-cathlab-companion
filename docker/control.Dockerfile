# Control Station (control plane + UI), core deps only. Simulators and LeRobot run as workers in their own envs
# (configs/workers.yaml; point CATHLAB_WORKERS at a file with reachable URLs).
#   docker build -f docker/control.Dockerfile -t cathlab-control .    # behind a proxy: --build-arg http(s)_proxy=...
#   docker run --rm -p 8700:8700 -v "$PWD/var:/app/var" cathlab-control
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev
EXPOSE 8700
CMD ["uv", "run", "--no-dev", "python", "-m", "orchestrator.control", "--host", "0.0.0.0", "--port", "8700", "--no-agent"]
