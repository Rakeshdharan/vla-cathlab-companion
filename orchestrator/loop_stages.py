"""Control-plane pipeline stages of the loop: register the run's evaluated checkpoint as a version, promote it.

Same registry, promotion rule and card fields as the MODELS tab (orchestrator/loop.py); these let a run go from
eval to production without leaving PIPELINE."""
import re
import subprocess
from pathlib import Path
from typing import Callable

from contracts import ModelCard
from orchestrator.store import Run
from serve.registry import MODEL_NAME, PromotionRule, Registry

VERSION_RE = re.compile(r"^v([1-9]\d*)$")


def _git_sha(repo_root: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def make_local_actions(pipeline, registry: Registry, repo_root: Path) -> dict[str, Callable[[Run, dict], dict]]:
    serve_yaml = Path(repo_root) / "configs/serve.yaml"
    rule = PromotionRule.from_yaml(serve_yaml) if serve_yaml.exists() else PromotionRule()

    def register(run: Run, params: dict) -> dict:
        ckpt = str(params.get("ckpt") or "").strip()
        if not ckpt:
            raise ValueError("register needs 'ckpt' (the HF repo or path that the eval stage evaluated)")
        engine = run.selection["simulator"]
        report = Path(pipeline.latest(run.id, "eval").result["out"]) / "eval_report.json"
        if not report.is_file():
            raise ValueError(f"the run's latest eval has no eval_report.json ({report})")
        existing = [int(m.group(1)) for v in registry.list_versions(MODEL_NAME)
                    if (m := VERSION_RE.match(v.card.version))]
        version = f"v{max(existing, default=0) + 1}"
        datasets = [d.strip() for d in str(params.get("datasets") or "").split(",") if d.strip()]
        card = ModelCard(model_name=MODEL_NAME, version=version, adapter="lerobot", ckpt_uri=ckpt,
                         dataset_versions=datasets, eval_report_path=str(report), git_sha=_git_sha(repo_root))
        registry.register(card, {engine: str(report)})
        return {"version": version, "eval_report": str(report),
                "message": f"registered {MODEL_NAME}@{version} ({ckpt}) with the {engine} eval\n"}

    def promote(run: Run, params: dict) -> dict:
        version = str(params.get("version") or "").strip() or pipeline.latest(run.id, "register").result["version"]
        d = registry.promote(MODEL_NAME, version, rule)
        out = {"version": version, **d.model_dump(mode="json")}
        if not d.promoted:
            return {**out, "error": f"{d.reason}\n"}
        return {**out, "message": f"{MODEL_NAME}@{version} is production: {d.reason}\n"}

    return {"register": register, "promote": promote}
