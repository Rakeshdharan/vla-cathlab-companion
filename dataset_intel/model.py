"""IntelReport: the JSON schema written by `python -m dataset_intel.report`."""
from typing import Literal
from pydantic import BaseModel, Field

IssueKind = Literal["nan", "frozen", "action_jump", "length_outlier", "gripper_never_closes", "idle_fraction",
                    "frame_count_mismatch"]


class DatasetInfo(BaseModel):
    root: str
    robot_type: str | None = None
    fps: int
    episodes: int
    frames: int


class Issue(BaseModel):
    episode: int
    kind: IssueKind
    detail: str
    severity: Literal["error", "warning"]


class Coverage(BaseModel):
    cells: dict[str, int] = Field(default_factory=dict)      # "object|layout|obj_color|table_color|source_sim" -> n
    empty_cells: list[str] = Field(default_factory=list)      # "object|layout|obj_color|table_color" with no episode
    train_cells: int = 0
    per_object: dict[str, int] = Field(default_factory=dict)
    per_source_sim: dict[str, int] = Field(default_factory=dict)


class Readiness(BaseModel):
    score: float
    valid_fraction: float
    coverage_fraction: float | None = None
    object_balance: float | None = None
    idle_penalty: float


class IntelReport(BaseModel):
    dataset: DatasetInfo
    issues: list[Issue] = Field(default_factory=list)
    coverage: Coverage | None = None
    readiness: Readiness
