from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from engines.registry import load_engines
from harness.__main__ import main
from harness.brief import brief_paths, build_brief
from harness.gates import GATE_IDS, load_gates
from harness.model import load_templates
from harness.server import create_agent_app
from tests.harness.conftest import make_harness

ROOT = Path(__file__).resolve().parents[2]
GATES = load_gates()
TEMPLATES = load_templates(ROOT / "harness/tasks")


def test_every_stage_has_a_valid_gate():
    assert set(GATES) == set(GATE_IDS)
    for g in GATES.values():
        assert g.contract and g.conformance and g.paths, g.id
        for rel in g.reference:
            assert (ROOT / rel).exists(), f"{g.id}: missing reference {rel}"


def test_templates_and_gates_point_at_each_other():
    for t in TEMPLATES.values():
        if t.id != "free_task":
            assert t.gate in GATES, f"template {t.id} has no valid gate"
    for g in GATES.values():
        for tid in g.templates:
            assert tid in TEMPLATES, f"gate {g.id} lists unknown template {tid}"


def test_brief_is_deterministic_and_complete():
    args = ("Add a Newton engine with the Panda and our table scene", ["eval", "engine"], ["newton", "robosuite"],
            GATES, TEMPLATES, load_engines())
    md = build_brief(*args)
    assert md == build_brief(*args)
    assert md.startswith("# Agent brief: Add a Newton engine")
    assert md.index("(`engine`") < md.index("(`eval`")                  # pipeline order, not request order
    assert GATES["engine"].contract[0] in md and GATES["engine"].conformance in md
    assert "python -m harness run add_engine" in md
    assert "`newton`: new engine" in md and "`robosuite` (" in md
    assert "Co-authored-by" in md and "cathlab_v1.yaml" in md


def test_brief_errors_and_paths():
    with pytest.raises(ValueError, match="unknown stage"):
        build_brief("x", ["warp"], [], GATES, TEMPLATES, {})
    with pytest.raises(ValueError, match="empty"):
        build_brief("  ", [], [], GATES, TEMPLATES, {})
    paths = brief_paths(["engine", "eval"], GATES)
    assert paths[0] == GATES["engine"].paths[0] and len(paths) == len(set(paths))


def test_brief_api_and_gates(tmp_path, repo):
    c = TestClient(create_agent_app(make_harness(tmp_path, repo)))
    assert len(c.get("/api/agent/gates").json()) == len(GATE_IDS)
    r = c.post("/api/agent/brief", json={"request": "Add ManiSkill", "stages": ["engine"], "engines": ["maniskill"]})
    assert r.status_code == 200 and "# Agent brief: Add ManiSkill" in r.json()["markdown"]
    assert r.json()["paths"] == GATES["engine"].paths
    assert c.post("/api/agent/brief", json={"request": "x", "stages": ["nope"]}).status_code == 400


def test_brief_cli_writes_file(tmp_path):
    out = tmp_path / "brief.md"           # `brief` reads templates and gates only; it never touches git
    rc = main(["--repo", str(ROOT), "--no-fetch", "brief", "Add a metric", "--stage", "eval", "--out", str(out)])
    assert rc == 0 and out.read_text().startswith("# Agent brief: Add a metric")
