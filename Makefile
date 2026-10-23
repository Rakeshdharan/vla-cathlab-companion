# CathLab Control Station: common commands. Run from the repo root.
# Native (no Docker) dev setup; see docs/INSTALL.md.

PORT ?= 8700
ROBOSUITE_ENV ?= robosuite
LEROBOT_ENV ?= lerobot
MANISKILL_ENV ?= maniskill

.PHONY: install test dev worker-robosuite worker-maniskill worker-lerobot agent demo-ui harness-list \
	conformance-robosuite conformance-maniskill

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
