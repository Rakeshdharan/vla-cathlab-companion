import json
import pytest
from fastapi.testclient import TestClient
from contracts import EpisodeResult, EvalReport
from orchestrator.control import create_control_app
from orchestrator.pipeline import PipelineError
from orchestrator.stages import STAGE_BY_ID
from orchestrator.worker import create_worker_app
from tests.orchestrator.test_pipeline import CATALOG, ROOT, router, wait_done


def write_report(var, engine, model, successes, n=60, p95=100.0, tier="in_distribution"):
    eps = [EpisodeResult(episode=i, seed=5000 + i % 4, object=["bottle", "vial", "syringe"][i % 3],
                         instruction="x", success=i < successes, failed_stage=None if i < successes else "grasp",
                         steps=50, latency_ms_p50=p95 / 2, latency_ms_p95=p95, scene_variant=tier) for i in range(n)]
    p = var / "evals" / engine / model / "eval_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(EvalReport(model_version=model, dataset_versions=[], backend=engine, episodes=eps).model_dump_json())
    return str(p)


@pytest.fixture
def env(tmp_path):
    var = tmp_path / "var"
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    (tmp_path / "workers.yaml").write_text('workers:\n  - {name: fake, url: "http://fake-worker"}\n')
    worker = TestClient(create_worker_app("fake", ["convert"], tmp_path / "worker"))
    app = create_control_app(ROOT, var_dir=var, catalog_path=tmp_path / "catalog.yaml",
                             workers_path=tmp_path / "workers.yaml", frontend_dir=tmp_path / "none",
                             engines_dir=ROOT / "tests/engines", worker_http=router({"http://fake-worker": worker}),
                             eval_roots=[var / "evals"])
    return TestClient(app), var, app


def test_evals_are_discovered(env):
    c, var, _ = env
    write_report(var, "robosuite", "smolvla", 23, p95=271.0)
    write_report(var, "maniskill", "smolvla", 0)
    evals = {(e["engine"], e["model"]): e for e in c.get("/api/evals").json()}
    assert evals[("robosuite", "smolvla")]["rate"] == round(23 / 60, 4)
    assert evals[("robosuite", "smolvla")]["p95_ms"] == pytest.approx(271.0)
    assert ("maniskill", "smolvla") in evals


def test_register_promote_rollback(env, tmp_path):
    c, var, _ = env
    a, b = write_report(var, "robosuite", "a", 20), write_report(var, "robosuite", "b", 24)
    v1 = c.post("/api/registry/register", json={"ckpt_uri": "u/a", "eval_reports": {"robosuite": a}}).json()
    assert v1["card"]["version"] == "v1" and v1["summary"]["robosuite"]["n"] == 60
    assert not v1["eval_reports"]["robosuite"].startswith("/") or "var" in v1["eval_reports"]["robosuite"]
    assert c.post("/api/registry/register", json={"ckpt_uri": "u/b", "eval_reports": {"robosuite": b}}).json()["card"]["version"] == "v2"
    assert c.post("/api/registry/register", json={"version": "v2", "ckpt_uri": "u", "eval_reports": {"robosuite": b}}).status_code == 409
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    assert c.post("/api/registry/register", json={"ckpt_uri": "u", "eval_reports": {"robosuite": str(outside)}}).status_code == 400
    assert c.post("/api/registry/register", json={"adapter": "lerobot", "ckpt_uri": "", "eval_reports": {"robosuite": a}}).status_code == 400
    assert c.post("/api/registry/register", json={"version": "v01", "ckpt_uri": "u", "eval_reports": {"robosuite": a}}).status_code == 400
    assert c.post("/api/registry/promote", json={"version": "v1"}).json()["promoted"]
    d = c.post("/api/registry/promote", json={"version": "v2"}).json()
    assert not d["promoted"] and "does not beat" in d["reason"]
    assert c.post("/api/registry/promote", json={"version": "v9"}).status_code == 404
    reg = c.get("/api/registry").json()
    assert reg["production"] == "v1" and [v["card"]["version"] for v in reg["versions"]] == ["v1", "v2"]
    assert reg["rule"]["latency_budget_ms_p95"] == 500.0
    assert c.post("/api/registry/rollback").json() == {"production": None}


def test_report_and_next_run(env):
    c, var, _ = env
    rs = write_report(var, "robosuite", "m", 10)
    r = c.post("/api/reports", json={"eval_reports": {"robosuite": rs}, "model": "m"}).json()
    assert r["markdown"].startswith("# Failure report: m") and r["next_run"]["cells"]
    assert c.get("/api/reports").json()[0]["id"] == r["id"]
    assert c.get(f"/api/reports/{r['id']}").json()["report"]["model"] == "m"
    assert c.get("/api/reports/..%2F..%2Fetc").status_code in (400, 404)
    run = c.post(f"/api/reports/{r['id']}/next-run", json={"note": "loop 1"}).json()
    assert run["selection"]["next_run"].endswith(f"{r['id']}/next_run.json") and run["note"] == "loop 1"
    total = sum(cell["k"] for cell in r["next_run"]["cells"])
    gen = {s["id"]: s for s in c.get(f"/api/runs/{run['id']}/stages").json()}["generate"]
    defaults = {p["name"]: p["default"] for p in gen["params"]}
    assert defaults["split"] == f"cells:{run['selection']['next_run']}" and defaults["n"] == total
    job = c.post(f"/api/runs/{run['id']}/stages/generate", json={"params": {}}).json()
    assert wait_done(c, job["id"])[0] == "done"
    argv = c.get(f"/api/jobs/{job['id']}").json()["result"]["argv"]
    assert argv[argv.index("--split") + 1].startswith("cells:") and argv[argv.index("--n") + 1] == str(total)


def test_registry_policies_resolve_in_the_pipeline(env):
    c, var, app = env
    a = write_report(var, "robosuite", "x", 50)
    c.post("/api/registry/register", json={"adapter": "expert", "eval_reports": {"robosuite": a}})
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    pipe, r = app.state.pipeline, app.state.store.get_run(run["id"])
    with pytest.raises(PipelineError) as e:
        pipe.effective_params(r, STAGE_BY_ID["eval"], {"policy": "registry:cathlab-pick@production"})
    assert e.value.status == 400
    c.post("/api/registry/promote", json={"version": "v1"})
    p = pipe.effective_params(r, STAGE_BY_ID["eval"], {"policy": "registry:cathlab-pick@production"})
    assert p["policy"] == "expert" and p["model-version"] == "cathlab-pick@production"
    assert pipe.effective_params(r, STAGE_BY_ID["rollout"], {"policy": "registry:cathlab-pick@v1"})["policy"] == "expert"
    assert pipe.effective_params(r, STAGE_BY_ID["generate"], {"n": 3}) == {"n": 3}
