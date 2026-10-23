"""Pipeline stages of a run: which engine CLI subcommand, on which worker, after what, with which params."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageParam:
    name: str                 # engine CLI flag without "--", e.g. "n", "max-seeds"
    type: str = "str"         # "str" | "int"
    default: Any = None
    help: str = ""


@dataclass(frozen=True)
class StageDef:
    id: str
    label: str
    cli: str                                   # engine CLI subcommand
    capability: str                            # manifest capability; "{expert}" = the run's expert id
    worker_capability: str | None = None       # None: the engine's worker; else any healthy worker advertising it
    needs: tuple = ()                          # stages whose latest job must be `done`
    inputs: dict = field(default_factory=dict)  # CLI flag -> stage whose output folder it receives
    params: tuple = ()

    def capability_for(self, selection: dict) -> str:
        return self.capability.format(expert=selection.get("expert") or "analytic")

    def param_dicts(self) -> list[dict]:
        return [asdict(p) for p in self.params]


STAGES: tuple = (
    StageDef("preview", "Scene preview", "preview", "preview",
             params=(StageParam("n", "int", 8, "train configs in the grid"),)),
    StageDef("check_expert", "Check expert", "check-expert", "expert.{expert}",
             params=(StageParam("n", "int", 5, "episodes per object"), StageParam("split", "str", "train"))),
    StageDef("generate", "Generate data", "generate", "generate",
             params=(StageParam("n", "int", 30, "configs (all objects, balanced)"),
                     StageParam("start", "int", 0, "first config of the window"), StageParam("split", "str", "train"))),
    StageDef("convert", "Convert to LeRobot", "convert", "convert", worker_capability="convert",
             needs=("generate",), inputs={"dataset": "generate"}),
    StageDef("eval", "Evaluate", "eval", "eval",
             params=(StageParam("policy", "str", "expert", "expert | random | lerobot:<hf-repo-or-path> | url:<policy-server-url> | registry:<name>@<version|production>"),
                     StageParam("model-version", "str", "", "label in the report"),
                     StageParam("tiers", "str", "in_distribution", "comma-separated; empty = all suite tiers"),
                     StageParam("max-seeds", "int", 1, "seeds per tier; empty = all"))),
    StageDef("rollout", "Rollout", "rollout", "rollout",
             params=(StageParam("policy", "str", "expert", "expert | random | lerobot:<hf-repo-or-path> | url:<policy-server-url> | registry:<name>@<version|production>"),
                     StageParam("object", "str", "", "default: first object of the run"),
                     StageParam("seed", "int", 5000), StageParam("instruction", "str", "", "override"))),
)
STAGE_BY_ID = {s.id: s for s in STAGES}


def stage_argv(stage: StageDef, params: dict) -> list[str]:
    """CLI flags for the user's params. Empty values are left out, so the engine's defaults apply."""
    known = {p.name: p for p in stage.params}
    argv: list[str] = []
    for name, value in params.items():
        if name not in known:
            raise ValueError(f"stage '{stage.id}' has no parameter '{name}'")
        if value is None or value == "":
            continue
        if known[name].type == "int":
            value = int(value)
        argv += [f"--{name}", str(value)]
    return argv
