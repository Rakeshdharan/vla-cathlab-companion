from pathlib import Path
from typing import Protocol
import yaml
from pydantic import BaseModel, model_validator


class HasObjects(Protocol):
    objects: list[str]


class ProcedureStep(BaseModel):
    step_id: str
    description: str
    tools: list[str]
    cue: str = ""


class ProcedureSpec(BaseModel):
    procedure_id: str
    steps: list[ProcedureStep]

    @model_validator(mode="after")
    def _unique_ids(self) -> "ProcedureSpec":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate step_id in procedure")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ProcedureSpec":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def index(self, step_id: str) -> int:
        for i, s in enumerate(self.steps):
            if s.step_id == step_id:
                return i
        raise KeyError(step_id)

    def next_tool(self, step_id: str) -> str | None:
        for step in self.steps[self.index(step_id) + 1 :]:
            if step.tools:
                return step.tools[0]
        return None

    def validate_against(self, task: HasObjects) -> None:
        unknown = sorted({t for s in self.steps for t in s.tools} - set(task.objects))
        if unknown:
            raise ValueError(f"procedure tools not in task objects: {', '.join(unknown)}")
