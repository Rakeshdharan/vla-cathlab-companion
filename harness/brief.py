"""Agent Brief: a deterministic markdown brief for any coding agent, from a request + the stage gates it touches."""
from engines.registry import EngineManifest
from harness.gates import GATE_IDS, Gate
from harness.model import TaskTemplate

RULES = (
    "Read AGENTS.md first and follow it.",
    "Branch from the tip of origin/main; small Conventional Commits.",
    "Commits are authored by the human only: no Co-authored-by trailers for AI agents.",
    "Never commit data, checkpoints, .env or secrets; outputs go under var/ or data/ (git-ignored).",
    "Never edit configs/variation/cathlab_v1.yaml (the published v1 dataset depends on it); add a new versioned spec.",
    "Add a row to CREDITS.md for every project, model, dataset or paper you use.",
    "Record decisions and evidence (check results) in docs/decisions.md.",
)


def _ordered(stage_ids: list[str], gates: dict[str, Gate]) -> list[Gate]:
    unknown = sorted(set(stage_ids) - set(gates))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected some of {list(GATE_IDS)}")
    order = {g: i for i, g in enumerate(GATE_IDS)}
    return [gates[s] for s in sorted(set(stage_ids), key=lambda s: (order.get(s, len(order)), s))]


def brief_paths(stage_ids: list[str], gates: dict[str, Gate]) -> list[str]:
    paths: list[str] = []
    for gate in _ordered(stage_ids, gates):
        paths += [p for p in gate.paths if p not in paths]
    return paths


def _capabilities(m: EngineManifest) -> list[tuple[str, str]]:
    flat = []
    for key, value in m.capabilities.items():
        flat += [(key, value)] if isinstance(value, str) else [(f"{key}.{k}", v) for k, v in value.items()]
    return flat


def build_brief(request: str, stage_ids: list[str], engine_ids: list[str], gates: dict[str, Gate],
                templates: dict[str, TaskTemplate], engines: dict[str, EngineManifest]) -> str:
    if not request.strip():
        raise ValueError("request is empty")
    selected = _ordered(stage_ids, gates)
    title = request.strip().splitlines()[0][:80]
    lines = [f"# Agent brief: {title}", "", "## Request", "", request.strip(), "", "## Repository rules", ""]
    lines += [f"- {rule}" for rule in RULES]
    for g in selected:
        lines += ["", f"## Stage gate: {g.title} (`{g.id}`, {g.status}, owner {g.owner})", "",
                  f"**Extension point:** {g.extension_point}", "", "**Contract:**", ""]
        lines += [f"- {c}" for c in g.contract]
        lines += ["", f"**Conformance (must pass):** `{g.conformance}`", "",
                  "**Allowed paths (suggested):** " + ", ".join(f"`{p}`" for p in g.paths)]
        if g.reference:
            lines += ["", "**Reference implementation:** " + ", ".join(f"`{r}`" for r in g.reference)]
        if g.context:
            lines += ["", "**Read first:** " + ", ".join(f"`{c}`" for c in g.context)]
        if g.conventions:
            lines += ["", "**Conventions:**", ""] + [f"- {c}" for c in g.conventions]
        matching = [templates[t] for t in g.templates if t in templates]
        if matching:
            lines += ["", "**Matching harness templates** (UI: AGENT tab):", ""]
            for t in matching:
                params = " ".join(f"--param {p.name}=..." for p in t.params if p.required)
                cmd = f"python -m harness run {t.id} {params}".rstrip()
                lines.append(f"- `{cmd}`: {t.title}")
    if engine_ids:
        lines += ["", "## Engines", ""]
        for eid in sorted(set(engine_ids)):
            m = engines.get(eid)
            if m is None:
                lines.append(f"- `{eid}`: new engine, not in engines/ yet. Follow the `engine` gate; copy engines/robosuite/.")
                continue
            pending = [f"{k} ({v})" for k, v in _capabilities(m) if v != "ready"]
            env = m.env.name or m.env.python or ""
            lines.append(f"- `{eid}` ({m.label}): env {m.env.kind} {env}, worker `{m.worker}`, io `{m.io}`; "
                         f"not ready: {', '.join(pending) or 'none'}")
    lines += ["", "## Commands", "", "```bash", "uv run pytest -q          # core tests stay green",
              "make dev                  # Control Station on http://localhost:8700"]
    lines += sorted({g.conformance for g in selected})
    lines += ["```", "", "## Done when", "",
              "- The conformance commands above pass and the new behaviour has tests.",
              "- Status flags change only for what now works (configs/catalog.yaml, engines/<id>/engine.yaml).",
              "- docs/decisions.md has a row with the evidence; CREDITS.md lists what you used.", ""]
    return "\n".join(lines)
