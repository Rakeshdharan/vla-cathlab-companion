# CathLab Control Station: common commands. Run from the repo root.
# Native (no Docker) dev setup; see docs/INSTALL.md.

PORT ?= 8700
ROBOSUITE_ENV ?= robosuite
LEROBOT_ENV ?= lerobot
MANISKILL_ENV ?= maniskill
GENESIS_PY ?= $(HOME)/venvs/genesis/bin/python
# cpu | gpu: one Genesis world is launch-bound; on a GPU busy with training the CPU backend is ~15x faster
GENESIS_BACKEND ?= cpu

.PHONY: install test dev worker-robosuite worker-maniskill worker-genesis worker-lerobot agent demo-ui harness-list \
	conformance-robosuite conformance-maniskill conformance-genesis policy-server demo-bootstrap

install:            ## core env (uv) with dev deps
	uv sync

test:               ## GPU-free test suite
	uv run pytest -q

dev:                ## control plane + UI + agent harness on http://localhost:$(PORT)
	uv run python -m orchestrator.control --port $(PORT)

worker-robosuite:   ## robosuite worker (scene, generate, eval-rollout) on :8711
	MUJOCO_GL=egl conda run -n $(ROBOSUITE_ENV) --no-capture-output python -m orchestrator.worker \
		--name robosuite --capabilities scene,generate,eval --port 8711

worker-maniskill:   ## ManiSkill3 worker (scene, generate, eval-rollout) on :8714
	conda run -n $(MANISKILL_ENV) --no-capture-output python -m orchestrator.worker \
		--name maniskill --capabilities scene,generate,eval --port 8714

worker-genesis:     ## Genesis worker (scene, generate, eval-rollout) on :8715
	CATHLAB_GENESIS_BACKEND=$(GENESIS_BACKEND) PYTHONPATH=. $(GENESIS_PY) -m orchestrator.worker \
		--name genesis --capabilities scene,generate,eval --port 8715

worker-lerobot:     ## lerobot worker (convert, intel, train, policy-server) on :8712
	conda run -n $(LEROBOT_ENV) --no-capture-output python -m orchestrator.worker \
		--name lerobot --capabilities convert,intel,train,policy_server --port 8712

agent:              ## standalone agent worker (harness API only) on :8790
	uv run python -m harness serve --port 8790

demo-ui:            ## UI with a throwaway repo and a fake agent (no Copilot login needed)
	uv run python scripts/demo_control_station.py

harness-list:       ## list agent task templates
	uv run python -m harness list

conformance-robosuite:  ## engine conformance test in the robosuite env
	MUJOCO_GL=egl conda run -n $(ROBOSUITE_ENV) --no-capture-output python -m pytest -q tests/engines

conformance-maniskill:  ## engine conformance test in the maniskill env
	MUJOCO_GL=egl conda run -n $(MANISKILL_ENV) --no-capture-output python -m pytest -q tests/engines -k maniskill

conformance-genesis:  ## engine + scene tests in the Genesis venv
	CATHLAB_GENESIS_BACKEND=$(GENESIS_BACKEND) PYTHONPATH=. timeout 1200 $(GENESIS_PY) -m pytest -q tests/engines -k genesis tests/sim/test_genesis_env.py

policy-server:      ## serve the production model on :8720 (lerobot env; needs a promoted lerobot version)
	conda run -n $(LEROBOT_ENV) --no-capture-output python -m serve.policy_server --registry cathlab-pick@production

demo-bootstrap:     ## rebuild var/registry + the production failure report from the committed eval reports
	uv run python -m scripts.demo_bootstrap
