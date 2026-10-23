from pathlib import Path
from typing import Literal, TypeVar
import yaml
from pydantic import BaseModel, Field

Stage = Literal["reach", "grasp", "lift", "transport", "place"]
STAGES: tuple[str, ...] = ("reach", "grasp", "lift", "transport", "place")
DataSource = Literal["provided", "domain_rand", "mimicgen", "scripted", "teleop", "cosmos"]
Backend = Literal["dummy", "robosuite", "isaac", "mujoco", "robosimstudio"]


class TaskSpec(BaseModel):
    task_id: str
    instruction_templates: list[str]
    objects: list[str]
    success_predicate: str
    stage_predicates: dict[Stage, str]

    def instructions(self, obj: str) -> list[str]:
        return [t.format(object=obj) for t in self.instruction_templates]


class SceneSpec(BaseModel):
    scene_id: str
    backend: Backend
    asset: str
    objects: dict[str, str]
    cameras: list[str]
    randomization: dict[str, list[float]] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    source: DataSource
    task_id: str
    scene_id: str
    n_episodes: int
    created_by: str
    parent_versions: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    notes: str = ""


M = TypeVar("M", bound=BaseModel)


def load_yaml(path: str | Path, model: type[M]) -> M:
    return model.model_validate(yaml.safe_load(Path(path).read_text()))


def save_yaml(obj: BaseModel, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj.model_dump(mode="json"), sort_keys=False))
