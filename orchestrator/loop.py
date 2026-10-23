"""Closed-loop API of the control plane: eval reports, model registry (versions, promote, rollback), failure reports
and the targeted next run. Mounted under /api by orchestrator/control.py."""
import json
import re
import subprocess
from pathlib import Path
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from contracts import EvalReport, ModelCard
from dataengine.variation import VariationSpec
from eval.metrics import latency_summary
from orchestrator.catalog import load_catalog, selection_problems
from orchestrator.store import Store, new_id
from report.failure_report import build_report, write_outputs
from serve.registry import MODEL_NAME, PromotionRule, Registry

VERSION_RE = re.compile(r"^v([1-9]\d*)$")
REPORT_ID_RE = re.compile(r"^rep-[A-Za-z0-9-]+$")


class RegisterBody(BaseModel):
    version: str | None = None
    adapter: str = "lerobot"
    ckpt_uri: str = ""
    dataset_versions: list[str] = Field(default_factory=list)
    eval_reports: dict[str, str]
    wandb_run: str | None = None


class PromoteBody(BaseModel):
    version: str
    force: bool = False


class ReportBody(BaseModel):
    eval_reports: dict[str, str]
    model: str
    k: int = Field(8, ge=1, le=100)
    top: int = Field(10, ge=1, le=100)
    spec: str = "configs/variation/cathlab_v1.yaml"


class NextRunBody(BaseModel):
    selection: dict | None = None
    note: str = ""


def summarize(path: Path) -> dict:
    r = EvalReport.load(path)
    lo, hi = r.wilson_ci()
    return {"engine": r.backend, "model": r.model_version, "n": len(r.episodes), "rate": round(r.success_rate(), 4),
            "ci": [round(lo, 4), round(hi, 4)], "p95_ms": round(latency_summary(r)["p95_ms"], 1)}


def create_loop_router(store: Store, registry: Registry, var_dir: Path, repo_root: Path, catalog_path: Path,
                       eval_roots: list[Path]) -> APIRouter:
    r = APIRouter()
    var_dir, repo_root = Path(var_dir), Path(repo_root)
    reports_dir = var_dir / "reports"
    serve_yaml = repo_root / "configs/serve.yaml"
    rule = PromotionRule.from_yaml(serve_yaml) if serve_yaml.exists() else PromotionRule()
    allowed = [repo_root.resolve(), var_dir.resolve()]

    def under_allowed(p: Path) -> bool:
        return any(a == p or a in p.parents for a in allowed)

    def checked(path: str, suffix: str = ".json") -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else repo_root / p).resolve()
        if p.suffix != suffix or not under_allowed(p) or not p.is_file():
            raise HTTPException(400, f"not a {suffix} file under the repo or var/: {path}")
        return p

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            return str(p.resolve())

    def safe_summary(path: Path) -> dict | None:
        try:
            return summarize(path)
        except (OSError, ValueError):
            return None

    def git_sha() -> str | None:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def version_view(m) -> dict:
        return {**m.model_dump(mode="json"),
                "summary": {e: safe_summary(registry.abs_path(p)) for e, p in m.eval_reports.items()}}

    # evals -----------------------------------------------------------------------------------------
    @r.get("/evals")
    def evals() -> list[dict]:
        out = []
        for root in eval_roots:
            root = Path(root)
            if not root.exists():
                continue
            for p in sorted(root.glob("**/eval_report.json")):
                s = safe_summary(p)
                if s is not None:
                    out.append({"path": rel(p), **s})
        return out

    # registry --------------------------------------------------------------------------------------
    @r.get("/registry")
    def get_registry() -> dict:
        prod = registry.production(MODEL_NAME)
        return {"name": MODEL_NAME, "production": prod.card.version if prod else None, "rule": rule.model_dump(),
                "versions": [version_view(m) for m in registry.list_versions(MODEL_NAME)],
                "history": registry.history(MODEL_NAME)}

    @r.post("/registry/register")
    def register(body: RegisterBody) -> dict:
        if body.adapter not in ("lerobot", "expert"):
            raise HTTPException(400, "adapter must be 'lerobot' or 'expert'")
        if body.adapter == "lerobot" and not body.ckpt_uri.strip():
            raise HTTPException(400, "ckpt_uri is required for a lerobot model (HF repo id or path)")
        if not body.eval_reports:
            raise HTTPException(400, "at least one eval report is required")
        evals_ = {engine: rel(checked(p)) for engine, p in body.eval_reports.items()}
        existing = [int(m.group(1)) for v in registry.list_versions(MODEL_NAME)
                    if (m := VERSION_RE.match(v.card.version))]
        version = body.version or f"v{max(existing, default=0) + 1}"
        if not VERSION_RE.match(version):
            raise HTTPException(400, "version must look like v<N>")
        card = ModelCard(model_name=MODEL_NAME, version=version, adapter=body.adapter, ckpt_uri=body.ckpt_uri.strip(),
                         dataset_versions=body.dataset_versions, eval_report_path="", wandb_run=body.wandb_run,
                         git_sha=git_sha())
        try:
            return version_view(registry.register(card, evals_))
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e

    @r.post("/registry/promote")
    def promote(body: PromoteBody) -> dict:
        try:
            return registry.promote(MODEL_NAME, body.version, rule, force=body.force).model_dump(mode="json")
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @r.post("/registry/rollback")
    def rollback() -> dict:
        m = registry.rollback(MODEL_NAME)
        return {"production": m.card.version if m else None}

    # failure reports + next run ----------------------------------------------------------------
    def report_dir(rid: str) -> Path:
        if not REPORT_ID_RE.match(rid) or not (reports_dir / rid / "report.json").is_file():
            raise HTTPException(404, f"unknown report {rid}")
        return reports_dir / rid

    def load_report(rid: str) -> dict:
        d = report_dir(rid)
        return {"id": rid, "report": json.loads((d / "report.json").read_text()),
                "markdown": (d / "report.md").read_text(), "next_run": json.loads((d / "next_run.json").read_text())}

    @r.post("/reports")
    def create_report(body: ReportBody) -> dict:
        evals_ = {engine: checked(p) for engine, p in body.eval_reports.items()}
        if not evals_:
            raise HTTPException(400, "at least one eval report is required")
        spec_path = checked(body.spec, ".yaml")
        try:
            reports = {e: EvalReport.load(p) for e, p in evals_.items()}
        except ValueError as e:
            raise HTTPException(400, f"invalid eval report: {e}") from e
        report = build_report(reports, VariationSpec.load(spec_path), rel(spec_path), body.model, body.top, body.k)
        report["eval_reports"] = {e: rel(p) for e, p in evals_.items()}
        rid = new_id("rep")
        write_outputs(report, reports_dir / rid)
        return load_report(rid)

    @r.get("/reports")
    def list_reports() -> list[dict]:
        out = []
        for d in reports_dir.glob("rep-*/report.json") if reports_dir.exists() else []:
            rep = json.loads(d.read_text())
            nr = json.loads((d.parent / "next_run.json").read_text())
            out.append({"id": d.parent.name, "model": rep["model"], "created": rep["created"],
                        "overall": rep["overall"], "cells": len(nr["cells"])})
        return sorted(out, key=lambda x: -x["created"])

    @r.get("/reports/{rid}")
    def get_report(rid: str) -> dict:
        return load_report(rid)

    @r.post("/reports/{rid}/next-run")
    def create_next_run(rid: str, body: NextRunBody) -> dict:
        path = (report_dir(rid) / "next_run.json").resolve()
        nr = json.loads(path.read_text())
        if not nr["cells"]:
            raise HTTPException(409, "the report has no train-split failing cells to target")
        catalog = load_catalog(catalog_path)
        selection = body.selection or dict(catalog.defaults)
        problems = selection_problems(catalog, selection)
        if problems:
            raise HTTPException(409, {"message": "selection is not runnable",
                                      "problems": [p.model_dump() for p in problems]})
        run = store.create_run({**selection, "next_run": str(path)},
                               body.note or f"next run from {rid} ({nr['model']}, {len(nr['cells'])} cells)")
        run_dir = var_dir / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run.model_dump(), sort_keys=False))
        return run.model_dump()

    return r
