from fastapi.testclient import TestClient

from orchestrator.control import create_control_app
from orchestrator.crossphysics import load_cross_physics
from tests.orchestrator.test_pipeline import ROOT


def test_committed_table_reshapes_into_rows_and_engine_columns():
    t = load_cross_physics(ROOT / "reports/robosuite/cross_physics_v1.json")
    assert t["engines"] == ["robosuite", "maniskill", "genesis"]
    rows = {r["model"]: r for r in t["rows"]}
    assert t["rows"][0]["model"] == "expert" and rows["expert"]["trained_on"] is None
    smol = rows["smolvla_panda_pick_v1"]
    assert smol["trained_on"] == "robosuite" and smol["cells"]["robosuite"]["in_domain"]
    assert round(smol["cells"]["robosuite"]["rate"], 3) == 0.383 and smol["cells"]["robosuite"]["n"] == 60
    assert not smol["cells"]["maniskill"]["in_domain"]
    assert round(rows["xvla_panda_genesis_v1"]["cells"]["robosuite"]["rate"], 3) == 0.183
    v2 = rows["xvla_panda_pick_v2"]
    assert t["rows"][-1]["model"] == "xvla_panda_pick_v2" and v2["trained_on"] == "robosuite+maniskill+genesis"
    assert all(v2["cells"][e]["in_domain"] for e in t["engines"])


def test_api_cross_physics(tmp_path):
    app = create_control_app(ROOT, var_dir=tmp_path / "var", catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend")
    body = TestClient(app).get("/api/cross-physics").json()
    assert body["source"] == "reports/robosuite/cross_physics_v1.json" and len(body["rows"]) >= 7
    assert load_cross_physics(tmp_path / "nope.json") == {"source": None, "engines": [], "rows": []}
