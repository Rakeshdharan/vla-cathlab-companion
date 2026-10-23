# Agent Harness: Team Guide

The harness turns a development task into a **reviewed git branch**, using a coding agent (GitHub Copilot CLI by
default). You pick a task template, fill a few fields, and the harness does the rest:

```
template + your fields → prompt → git worktree on agent/<task>-<id> (from origin/main)
  → agent edits files → path guard (only allowed paths) → acceptance checks → commit (your git identity)
  → you review the DIFF → PUSH BRANCH (then open a PR) or DISCARD
```

Your working folder and `main` are never touched. Every job keeps its prompt, full log and diff in `var/agent/<id>/`.

## Use it from the UI

1. `make dev` and open http://localhost:8700.
2. **PROJECT** tab: anything that isn't ready shows **Build with agent**, and sections with an add-task show
   **+ Add … with agent**.
3. Or go straight to **AGENT** and pick a template on the left.
4. Fill the fields, add extra instructions if needed, choose the agent, then **Start agent**.
5. Watch the live log. When the state is `ready_for_review`:
   - **DIFF** shows the full change;
   - **PROMPT** shows exactly what the agent was told;
   - **PUSH BRANCH** pushes `agent/<task>-<id>`; open a PR on GitHub from it;
   - **DISCARD** deletes the worktree and branch.

## Use it from the terminal

```bash
uv run python -m harness list
uv run python -m harness run add_object --param name=guidewire_hoop --param label="Guidewire hoop" \
    --param shape=cylinder --param radius=0.06 --param length=0.02 --param upright=false --param mass=0.05
uv run python -m harness diff <job_id>
uv run python -m harness push <job_id>        # or: discard <job_id>
```

## Templates

| Template | Use it to | Checks |
|---|---|---|
| `add_object` | Add a pickable object (new versioned spec `cathlab_v2`, scene, task, catalog) | core tests, robosuite tests, expert ≥ 80 % on the new object |
| `extend_dataset` | Generate more episodes for a window of configs, convert, preview, push to HF | dataset + manifest exist |
| `add_variation_axis` | Add a randomization axis (camera jitter, distractors, …) | core + robosuite tests, expert ≥ 80 % |
| `add_sim_backend` | Build an Isaac or MuJoCo backend against `ExpertSim` / `SimBackend` | core tests |
| `add_policy_adapter` | Wrap a new model (adapter + training recipe) | core tests |
| `add_expert` | Add a grasp source such as GraspGen | core + robosuite tests |
| `free_task` | Anything else: your goal, your allowed paths, your check | your command |

To add a template, drop a YAML file in `harness/tasks/` (the file name must equal its `id`).
`tests/harness/test_templates.py` checks that it loads and renders.

## Rules

- Review every diff before pushing. The agent is fast, not infallible.
- Templates never edit `configs/variation/cathlab_v1.yaml`: the published v1 dataset depends on it.
- Commits carry **your** name; the repo hook strips AI co-author trailers (`git config core.hooksPath .githooks`).
- Jobs that need robosuite run checks in the `robosuite` conda env, so set it up first (docs/INSTALL.md).
- Agents: `configs/agents.yaml` (`copilot` default; `claude`, `codex` optional). Log in to the agent CLI on the host
  before starting the control plane.

## Safety

- Agents run with an allow-listed environment (`env_allow` in `configs/agents.yaml`). Tokens like `HF_TOKEN` and `WANDB_API_KEY` are not visible to agents.
- Pushes from inside a job worktree are disabled via worktree-scoped `remote.origin.pushurl`. A job fails if the agent commits on its own.
- The first job sets `extensions.worktreeConfig=true` in the repo's shared git config. This is required by git for per-worktree settings and is harmless for other checkouts.
- Stuck jobs after a crash: run `python -m harness discard <id> --force`, or send `?force=true` to the API `/discard` endpoint.
- Template identifier params are pattern-checked and check commands shell-quote parameter values.
