"""Stage gates: for each pipeline stage, where an extension goes, the contract it meets and how it is checked."""
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field

GATES_DIR = Path(__file__).resolve().parent / "gates"
GATE_IDS = ("engine", "scene", "objects", "expert", "data_method", "convert", "intel", "train", "eval", "rollout",
            "assist", "report")


class Gate(BaseModel):
    id: str
    title: str
    status: Literal["ready", "partial", "planned"]
    owner: str
    extension_point: str
    contract: list[str]
    conformance: str
    paths: list[str]                      # suggested allowed paths for an agent job on this stage
    reference: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    templates: list[str] = Field(default_factory=list)
    conventions: list[str] = Field(default_factory=list)


def load_gates(gates_dir: str | Path = GATES_DIR) -> dict[str, Gate]:
    gates: dict[str, Gate] = {}
    for p in sorted(Path(gates_dir).glob("*.yaml")):
        gate = Gate.model_validate(yaml.safe_load(p.read_text()))
        if gate.id != p.stem:
            raise ValueError(f"{p}: id '{gate.id}' must equal the file name")
        gates[gate.id] = gate
    return gates
