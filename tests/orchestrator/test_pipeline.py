import time
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from orchestrator.control import create_control_app
from orchestrator.stages import STAGE_BY_ID, stage_argv
from orchestrator.worker import create_worker_app

ROOT = Path(__file__).resolve().parents[2]
CATALOG = """catalog_version: 1
sections:
  simulator: {title: Simulator, items: [{id: fake, label: Fake, status: ready}]}
  expert: {title: Expert, items: [{id: analytic, label: Analytic, status: ready}]}
  objects: {title: Objects, multi: true, items: [{id: vial, label: Vial, status: ready}]}
defaults: {simulator: fake, expert: analytic, objects: [vial]}
"""


def router(clients: dict):
    def http(method, url, timeout=None, **kw):
        for base, client in clients.items():
            if url.startswith(base):
                return client.request(method, url[len(base):], **kw)
        raise httpx.ConnectError(f"no route to {url}")
    return http


def control(tmp_path, worker_online=True):
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    (tmp_path / "workers.yaml").write_text('workers:\n  - {name: fake, url: "http://fake-worker"}\n')
    clients = {}
    if worker_online:
        clients["http://fake-worker"] = TestClient(create_worker_app("fake", ["convert"], tmp_path / "worker"))
    app = create_control_app(ROOT, var_dir=tmp_path / "var", catalog_path=tmp_path / "catalog.yaml",
                             workers_path=tmp_path / "workers.yaml", frontend_dir=tmp_path / "none",
                             engines_dir=ROOT / "tests/engines", worker_http=router(clients))
    return TestClient(app)


def wait_done(c, job_id, timeout=30):
    end, state, text, offset = time.time() + timeout, "", "", 0
    while time.time() < end:
        r = c.get(f"/api/jobs/{job_id}/log", params={"offset": offset}).json()
        text, offset, state = text + r["text"], r["offset"], r["state"]
        if state not in ("queued", "running"):
            break
        time.sleep(0.1)
    return state, text


def stages(c, run_id):
    return {s["id"]: s for s in c.get(f"/api/runs/{run_id}/stages").json()}


def test_stage_argv_skips_empty_and_validates():
    gen = STAGE_BY_ID["generate"]
    assert stage_argv(gen, {"n": "6", "start": "", "split": "train"}) == ["--n", "6", "--split", "train"]
    with pytest.raises(ValueError, match="no parameter 'bogus'"):
        stage_argv(gen, {"bogus": 1})
    with pytest.raises(ValueError):
        stage_argv(gen, {"n": "six"})


def test_pipeline_end_to_end_with_fake_engine(tmp_path):
    c = control(tmp_path)
    assert [e["id"] for e in c.get("/api/engines").json()] == ["fake"]
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    st = stages(c, run["id"])
    assert st["preview"]["enabled"] and st["preview"]["worker"] == "fake"
    assert st["convert"]["reason"] == "needs 'generate' to finish first"
    assert st["eval"]["reason"] == "fake: 'eval' is planned" and not st["eval"]["enabled"]

    job = c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {"n": 3}}).json()
    state, text = wait_done(c, job["id"])
    assert state == "done" and "preview n=3 objects=vial" in text
    assert [f["path"] for f in c.get(f"/api/jobs/{job['id']}/files").json()] == ["grid.txt"]
    assert c.get(f"/api/jobs/{job['id']}/file", params={"path": "grid.txt"}).text == "n=3"
    assert c.get(f"/api/jobs/{job['id']}/file", params={"path": "../../run.yaml"}).status_code == 400

    gen = c.post(f"/api/runs/{run['id']}/stages/generate", json={"params": {"n": 2}}).json()
    assert wait_done(c, gen["id"])[0] == "done"
    assert stages(c, run["id"])["convert"]["enabled"]
    conv = c.post(f"/api/runs/{run['id']}/stages/convert", json={"params": {}}).json()
    assert wait_done(c, conv["id"])[0] == "done"
    lerobot = c.get(f"/api/jobs/{conv['id']}/file", params={"path": "lerobot.txt"}).text
    assert lerobot == c.get(f"/api/jobs/{gen['id']}").json()["result"]["out"]

    assert c.post(f"/api/runs/{run['id']}/stages/eval", json={"params": {}}).status_code == 409
    assert c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {"bogus": 1}}).status_code == 400
    assert c.post(f"/api/runs/{run['id']}/stages/nope", json={"params": {}}).status_code == 404


def test_offline_worker_blocks_stages(tmp_path):
    c = control(tmp_path, worker_online=False)
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    st = stages(c, run["id"])
    assert st["preview"]["reason"] == "worker 'fake' is offline"
    assert st["convert"]["reason"] == "no online worker with capability 'convert'"
    assert c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {}}).status_code == 409


def test_cancel_running_stage(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    job = c.post(f"/api/runs/{run['id']}/stages/preview", json={"params": {}}).json()
    assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    assert wait_done(c, job["id"])[0] in ("cancelled", "done")
