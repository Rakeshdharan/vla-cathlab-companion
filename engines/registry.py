"""Engine plugin manifests (engines/<id>/engine.yaml): load, validate, query capabilities. No simulator imports."""
from pathlib import Path
from typing import Literal, Union
import yaml
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
ENGINES_DIR = ROOT / "engines"
Status = Literal["ready", "in_progress", "planned"]
STATUS_RANK = {"planned": 0, "in_progress": 1, "ready": 2}


class IOProfile(BaseModel):
    id: str
    action_dim: int
    fps: int
    obs_shapes: dict[str, tuple[int, ...]] = Field(default_factory=dict)  # empty: not checked by conformance
    description: str = ""


IO_PROFILES: dict[str, IOProfile] = {
    "panda_libero": IOProfile(
        id="panda_libero", action_dim=7, fps=20,
        obs_shapes={"state": (8,), "agentview": (256, 256, 3), "eye_in_hand": (256, 256, 3)},
        description="7-D normalized EEF delta [dpos, drot, gripper], 8-D state, agentview + eye_in_hand 256², 20 Hz"),
    "fr3_duo_16d": IOProfile(
        id="fr3_duo_16d", action_dim=16, fps=30,
        description="Siemens FR3 Duo schema: 16-D joint targets, 30 Hz (provided datasets; FR3 Duo MuJoCo/Newton track)"),
}


class EngineEnv(BaseModel):
    kind: Literal["conda", "venv", "command"]
    name: str | None = None     # conda env name or venv path
    python: str | None = None   # interpreter command for kind=command, e.g. "${ENGINE_PYTHON}"


class Credit(BaseModel):
    name: str
    url: str = ""
    license: str = ""


class EngineManifest(BaseModel):
    id: str
    label: str
    env: EngineEnv
    worker: str                  # worker name in configs/workers.yaml that runs this engine's stages
    embodiments: list[str]
    io: str
    variation: str               # default variation spec, repo-relative
    task: str                    # task spec, repo-relative
    scene_asset_env: str | None = None
    capabilities: dict[str, Union[Status, dict[str, Status]]]
    entry: str                   # python -m <entry> <stage> ...
    backend: str | None = None   # "module:Class" implementing contracts.SimBackend (conformance test)
    credits: list[Credit] = Field(default_factory=list)

    @field_validator("io")
    @classmethod
    def _known_io(cls, v: str) -> str:
        if v not in IO_PROFILES:
            raise ValueError(f"unknown io profile '{v}'; expected one of {sorted(IO_PROFILES)}")
        return v

    @property
    def io_profile(self) -> IOProfile:
        return IO_PROFILES[self.io]

    def status(self, capability: str) -> Status:
        """`generate` or dotted `expert.analytic`. A group without a key reports its best child. Unknown: planned."""
        key, _, sub = capability.partition(".")
        value = self.capabilities.get(key)
        if value is None:
            return "planned"
        if isinstance(value, str):
            return value
        if sub:
            return value.get(sub, "planned")
        return max(value.values(), key=STATUS_RANK.__getitem__, default="planned")


class EngineError(ValueError):
    pass


def load_manifest(path: str | Path) -> EngineManifest:
    path = Path(path)
    manifest = EngineManifest.model_validate(yaml.safe_load(path.read_text()))
    if manifest.id != path.parent.name:
        raise EngineError(f"{path}: id '{manifest.id}' must equal its folder name '{path.parent.name}'")
    return manifest


def load_engines(engines_dir: str | Path = ENGINES_DIR) -> dict[str, EngineManifest]:
    manifests = (load_manifest(p) for p in sorted(Path(engines_dir).glob("*/engine.yaml")))
    return {m.id: m for m in manifests}


def get_engine(engine_id: str, engines_dir: str | Path = ENGINES_DIR) -> EngineManifest:
    path = Path(engines_dir) / engine_id / "engine.yaml"
    if not path.exists():
        raise EngineError(f"no engine plugin '{engine_id}' (expected {path})")
    return load_manifest(path)
