# Installation

Two ways to run the CathLab Control Station:

| | Native (today) | Docker |
|---|---|---|
| Control plane + UI + agent harness | `uv` env on the host | `control` + `frontend` containers; the agent harness stays on the host |
| robosuite worker | conda env `robosuite` | `worker-robosuite` image |
| lerobot worker (convert, train, policy server) | conda env `lerobot` (laptop or GPU box) | `worker-lerobot` image (GPU) |

The Docker images and `compose.yaml` arrive with the Docker task; this page then gets a Docker section.

## 1. Prerequisites

- Linux, Python 3.10+, git, [uv](https://docs.astral.sh/uv/), conda (Miniforge recommended).
- For the agent harness: an agent CLI logged in on the host. Default: [GitHub Copilot CLI](https://docs.github.com/copilot/how-tos/use-copilot-agents/use-copilot-cli) (`copilot` then `/login`). Optional: Claude Code or Codex (see `configs/agents.yaml`).
- For rendering in robosuite: an EGL-capable GPU driver (`MUJOCO_GL=egl`), or `MUJOCO_GL=osmesa` without a GPU.

## 2. Clone and core environment

```bash
git clone https://github.com/Rakeshdharan/vla-cathlab-companion && cd vla-cathlab-companion
git config core.hooksPath .githooks        # strips AI co-author trailers from commits
make install                               # = uv sync (control plane, harness, tests)
make test                                  # GPU-free tests must pass
```

## 3. Simulator and training environments

```bash
# robosuite worker env (Python 3.10). Keep these exact versions.
conda create -y -n robosuite python=3.10
conda run -n robosuite pip install robosuite==1.5.2 "mujoco==3.3.7" "numpy<2" pydantic pyyaml imageio pytest \
    fastapi uvicorn httpx

# lerobot worker env (Python 3.12, CUDA GPU for training)
conda create -y -n lerobot python=3.12
conda run -n lerobot pip install "lerobot>=0.5,<0.6" pytest fastapi uvicorn httpx
```

Check the robosuite env: `MUJOCO_GL=egl conda run -n robosuite python -m pytest -q tests/sim`.

## 4. Start the stack (four terminals, or tmux)

```bash
make dev                # control plane + UI + agent harness  -> http://localhost:8700
make worker-robosuite   # :8711
make worker-maniskill   # :8714 (conda env `maniskill`: see sim/maniskill/README.md, plus `pip install fastapi uvicorn httpx`)
make worker-genesis     # :8715 (uv venv ~/venvs/genesis: see sim/genesis/README.md, plus `pip install fastapi uvicorn httpx`)
make worker-lerobot     # :8712 (or start it on the GPU box, see below)
```

The worker lamps in the UI header turn green when a worker answers on its URL (`configs/workers.yaml`).

The **PIPELINE** tab runs a run's stages on these workers: `make worker-robosuite` (or `make worker-maniskill`) for
the engine's `preview`/`check-expert`/`generate`/`eval`/`rollout` stages, `make worker-lerobot` for `convert`. A
stage that can't start shows *why* (no online worker with the needed capability, or a prior stage hasn't finished).

**Remote GPU box:** clone the repo there, create the `lerobot` env, and set a shared token on both machines. A remote
worker must see the **same repo path and `var/`** (same clone location, or a shared mount), because stage commands
use absolute paths under the control plane's `var/runs/`.

```bash
# On the control machine (where you run make dev):
export CATHLAB_WORKER_TOKEN=<shared-secret>
make dev

# On the GPU box:
export CATHLAB_WORKER_TOKEN=<shared-secret>
conda run -n lerobot python -m orchestrator.worker --name lerobot --capabilities convert,intel,train,policy_server \
    --host 0.0.0.0 --port 8712 --token "$CATHLAB_WORKER_TOKEN"
```

The control plane sends the token automatically when the env var is set where `make dev` runs.

Servers only answer requests addressed to `localhost`/`127.0.0.1` (this blocks DNS-rebinding attacks from web pages).
Workers started with a token accept any host name, because the token protects them. If you open the UI or a tokenless
service by a LAN name or IP, list it: `export CATHLAB_ALLOWED_HOSTS=my-pc,10.0.0.5`.
Then point the `lerobot` entry in `configs/workers.yaml` (or your own file passed with `CATHLAB_WORKERS=...`) at
`http://<gpu-box>:8712`. Use an SSH tunnel (`ssh -N -L 8712:localhost:8712 <gpu-box>`) if the port isn't reachable.

## 5. Try the UI without touching the repo

`make demo-ui` starts the UI against a throwaway git repo with a fake agent. Try AGENT → Free task with paths `configs/`
and check `test -f configs/new.txt`, then Start agent → DIFF → PUSH BRANCH.

## Ports

| Port | Service |
|---|---|
| 8700 | control plane + UI (+ agent harness in dev) |
| 8711 | robosuite worker |
| 8714 | ManiSkill3 worker |
| 8715 | Genesis worker |
| 8712 | lerobot worker |
| 8790 | standalone agent worker (`make agent`) |
