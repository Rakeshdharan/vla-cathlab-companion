from fastapi.testclient import TestClient
from orchestrator.control import create_control_app
from tests.orchestrator.test_pipeline import CATALOG, ROOT


def client(tmp_path):
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    (tmp_path / "workers.yaml").write_text("workers: []\n")
    return TestClient(create_control_app(ROOT, var_dir=tmp_path / "var", catalog_path=tmp_path / "catalog.yaml",
                                         workers_path=tmp_path / "workers.yaml", frontend_dir=tmp_path / "none",
                                         engines_dir=ROOT / "tests/engines"))


def test_procedures_and_proposals(tmp_path):
    c = client(tmp_path)
    procs = {p["id"]: p for p in c.get("/api/assist/procedures").json()}
    steps = [s["step_id"] for s in procs["coronary_angiography"]["steps"]]
    assert steps[0] == "access_prep" and "contrast_injection" in steps
    now = c.post("/api/assist/propose", json={"procedure": "coronary_angiography", "step_id": "access_prep"}).json()
    assert now["object"] == "vial" and now["reason"] == "needed now"
    assert now["question"] == "Shall I hand you the vial?" and "vial" in now["instruction"]
    nxt = c.post("/api/assist/propose", json={"procedure": "coronary_angiography", "step_id": "catheter_advance"}).json()
    assert nxt["object"] == "bottle" and nxt["reason"] == "needed for the next step 'contrast_injection'"
    last = c.post("/api/assist/propose", json={"procedure": "coronary_angiography", "step_id": "closure"}).json()
    assert last["object"] is None
    assert c.post("/api/assist/propose", json={"procedure": "nope", "step_id": "x"}).status_code == 404
    assert c.post("/api/assist/propose", json={"procedure": "coronary_angiography", "step_id": "x"}).status_code == 404


def test_handover_starts_a_rollout_or_explains_why_not(tmp_path):
    c = client(tmp_path)
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    r = c.post("/api/assist/handover", json={"run_id": run["id"], "object": "vial", "instruction": "Pass the vial",
                                             "policy": "expert"})
    assert r.status_code == 409 and "rollout" in r.json()["detail"]          # fake engine has no rollout capability
    assert c.post("/api/assist/handover", json={"run_id": "run-x", "object": "vial", "instruction": "x"}).status_code == 404
