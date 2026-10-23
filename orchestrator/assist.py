"""ASSIST: the awareness procedure model proposes the object the clinician needs; on YES the production model hands it
over in simulation (a rollout stage with the instruction). The current step is picked in the UI today; the VLM step
tracker (awareness.engine.AwarenessEngine) can drive the same endpoint later."""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from awareness.procedure import ProcedureSpec
from contracts import TaskSpec, load_yaml
from orchestrator.pipeline import Pipeline, PipelineError
from orchestrator.store import Store


class ProposeBody(BaseModel):
    procedure: str
    step_id: str


class HandoverBody(BaseModel):
    run_id: str
    object: str
    instruction: str
    policy: str = "registry:cathlab-pick@production"
    seed: int = 5000


def create_assist_router(store: Store, pipeline: Pipeline, repo_root: Path) -> APIRouter:
    r = APIRouter()
    proc_dir = Path(repo_root) / "configs/procedures"
    task = load_yaml(Path(repo_root) / "configs/tasks/cathlab_pick.yaml", TaskSpec)

    def procedures() -> dict[str, ProcedureSpec]:
        out = {}
        for p in sorted(proc_dir.glob("*.yaml")):
            spec = ProcedureSpec.from_yaml(p)
            spec.validate_against(task)
            out[spec.procedure_id] = spec
        return out

    @r.get("/procedures")
    def list_procedures() -> list[dict]:
        return [{"id": pid, "steps": [s.model_dump() for s in spec.steps]} for pid, spec in procedures().items()]

    @r.post("/propose")
    def propose(body: ProposeBody) -> dict:
        spec = procedures().get(body.procedure)
        if spec is None:
            raise HTTPException(404, f"unknown procedure {body.procedure}")
        try:
            step = spec.steps[spec.index(body.step_id)]
        except KeyError as e:
            raise HTTPException(404, f"unknown step {body.step_id}") from e
        if step.tools:
            obj, reason = step.tools[0], "needed now"
        else:
            obj = spec.next_tool(step.step_id)
            later = next((s for s in spec.steps[spec.index(step.step_id) + 1:] if obj in s.tools), None)
            reason = f"needed for the next step '{later.step_id}'" if later else "nothing to hand over"
        if obj is None:
            return {"procedure": body.procedure, "step_id": step.step_id, "object": None, "question": None,
                    "instruction": None, "reason": "nothing to hand over for the rest of the procedure"}
        return {"procedure": body.procedure, "step_id": step.step_id, "object": obj,
                "question": f"Shall I hand you the {obj}?", "instruction": task.instructions(obj)[0], "reason": reason}

    @r.post("/handover")
    def handover(body: HandoverBody) -> dict:
        run = store.get_run(body.run_id)
        if run is None:
            raise HTTPException(404, f"unknown run {body.run_id}")
        params = {"policy": body.policy, "object": body.object, "instruction": body.instruction, "seed": body.seed}
        try:
            return pipeline.start(run, "rollout", params).model_dump()
        except PipelineError as e:
            raise HTTPException(e.status, str(e)) from e

    return r
