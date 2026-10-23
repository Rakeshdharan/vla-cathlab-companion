"""Project catalog: every option the Control Station offers, with a readiness status."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

Status = Literal["ready", "in_progress", "planned"]
ProblemKind = Literal["unknown_section", "missing", "unknown_item", "not_ready", "requires", "min", "type"]


class CatalogItem(BaseModel):
    id: str
    label: str
    status: Status
    notes: str = ""
    harness_task: str | None = None
    requires: dict[str, list[str]] = Field(default_factory=dict)


class CatalogSection(BaseModel):
    title: str
    multi: bool = False
    min: int = 1
    add_task: str | None = None
    items: list[CatalogItem]

    def item(self, item_id: str) -> CatalogItem | None:
        return next((i for i in self.items if i.id == item_id), None)


class Catalog(BaseModel):
    catalog_version: int
    sections: dict[str, CatalogSection]
    defaults: dict[str, str | list[str]]


class Problem(BaseModel):
    section: str
    item: str | None = None
    kind: ProblemKind
    message: str
    harness_task: str | None = None


def load_catalog(path: str | Path) -> Catalog:
    return Catalog.model_validate(yaml.safe_load(Path(path).read_text()))


def selection_problems(catalog: Catalog, selection: dict) -> list[Problem]:
    """Everything that stops this selection from running. Empty list = runnable."""
    problems: list[Problem] = []
    for key in selection:
        if key not in catalog.sections:
            problems.append(Problem(section=key, kind="unknown_section", message=f"unknown section '{key}'"))
    for name, section in catalog.sections.items():
        if name not in selection:
            problems.append(Problem(section=name, kind="missing", message=f"{section.title}: nothing selected"))
            continue
        value = selection[name]
        if section.multi != isinstance(value, list):
            kind = "a list" if section.multi else "a single id"
            problems.append(Problem(section=name, kind="type", message=f"{section.title}: expected {kind}"))
            continue
        ids = value if section.multi else [value]
        if section.multi and len(ids) < section.min:
            problems.append(Problem(section=name, kind="min", harness_task=section.add_task,
                                    message=f"{section.title}: select at least {section.min}"))
        for item_id in ids:
            item = section.item(item_id)
            if item is None:
                problems.append(Problem(section=name, item=item_id, kind="unknown_item", harness_task=section.add_task,
                                        message=f"{section.title}: unknown item '{item_id}'"))
                continue
            if item.status != "ready":
                problems.append(Problem(section=name, item=item_id, kind="not_ready", harness_task=item.harness_task,
                                        message=f"{item.label} is {item.status.replace('_', ' ')}"))
            for other, allowed in item.requires.items():
                chosen = selection.get(other)
                chosen_ids = chosen if isinstance(chosen, list) else [chosen]
                if not any(c in allowed for c in chosen_ids):
                    problems.append(Problem(section=name, item=item_id, kind="requires",
                                            message=f"{item.label} requires {other} in {allowed}"))
    return problems
