"""Harness task templates: typed parameters, context files, prompt, allowed paths, acceptance checks."""
from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field


class Param(BaseModel):
    name: str
    type: Literal["str", "int", "float", "bool", "choice", "text"] = "str"
    default: Any = None
    choices: list[str] = Field(default_factory=list)
    required: bool = True
    help: str = ""


class Check(BaseModel):
    name: str
    cmd: str
    timeout_s: float = 900


class TaskTemplate(BaseModel):
    id: str
    title: str
    description: str
    catalog_target: str = ""
    params: list[Param] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    prompt: str
    allowed_paths: list[str]
    acceptance: list[Check] = Field(default_factory=list)
    timeout_min: float = 30


class ParamError(ValueError):
    pass


def load_templates(directory: str | Path) -> dict[str, TaskTemplate]:
    out = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        t = TaskTemplate.model_validate(yaml.safe_load(path.read_text()))
        if t.id != path.stem:
            raise ValueError(f"{path.name}: id '{t.id}' must equal the file name")
        out[t.id] = t
    return out


def _coerce(p: Param, raw: Any) -> Any:
    if p.type in ("str", "text"):
        return str(raw)
    if p.type == "int":
        return int(raw)
    if p.type == "float":
        return float(raw)
    if p.type == "bool":
        return raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "yes", "on")
    if str(raw) not in p.choices:
        raise ParamError(f"{p.name}: '{raw}' not in {p.choices}")
    return str(raw)


def validate_params(template: TaskTemplate, params: dict) -> dict:
    """Coerce to declared types, apply defaults, reject unknown/missing/invalid values."""
    known = {p.name: p for p in template.params}
    unknown = set(params) - set(known)
    if unknown:
        raise ParamError(f"unknown parameter(s): {sorted(unknown)}")
    out = {}
    for p in template.params:
        raw = params.get(p.name, p.default)
        if raw is None or raw == "":
            if p.required:
                raise ParamError(f"missing required parameter '{p.name}'")
            out[p.name] = ""
            continue
        try:
            out[p.name] = _coerce(p, raw)
        except (TypeError, ValueError) as e:
            raise ParamError(f"{p.name}: {e}") from e
    return out


def allowed_for(template: TaskTemplate, params: dict) -> list[str]:
    """Allowed path patterns with {param} placeholders filled; comma-separated values expand to several."""
    out = []
    for pattern in template.allowed_paths:
        out += [x.strip() for x in pattern.format(**params).split(",") if x.strip()]
    return out


def checks_for(template: TaskTemplate, params: dict) -> list[Check]:
    return [Check(name=c.name, cmd=c.cmd.format(**params), timeout_s=c.timeout_s) for c in template.acceptance]


RULES = """You are working in a git worktree of the CathLab Companion repo, on a fresh branch.
Rules:
- Read AGENTS.md and the context files listed below before changing anything.
- Only create or edit files matching the allowed paths below. Any other change fails the job.
- Do NOT run `git commit` or `git push`; the harness commits and a human reviews the diff.
- Never print, commit or hard-code secrets (.env, tokens, keys).
- Keep tests runnable without GPU or network unless the acceptance checks say otherwise.
- Finish by running the acceptance checks yourself and fixing failures."""


def render_prompt(template: TaskTemplate, params: dict, extra: str = "") -> str:
    body = template.prompt.format(**params)
    parts = [
        f"# Task: {template.title}",
        RULES,
        "## Context files (read first)\n" + "\n".join(f"- {c}" for c in template.context),
        "## Parameters\n" + "\n".join(f"- {k}: {v}" for k, v in params.items()) if params else "",
        "## Instructions\n" + body.strip(),
        "## Allowed paths\n" + "\n".join(f"- {a}" for a in allowed_for(template, params)),
        "## Acceptance checks (must pass)\n" + "\n".join(f"- `{c.cmd}`" for c in checks_for(template, params)),
        f"## Additional instructions from the user\n{extra.strip()}" if extra.strip() else "",
    ]
    return "\n\n".join(p for p in parts if p) + "\n"
