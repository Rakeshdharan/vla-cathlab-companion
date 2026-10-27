import json

from contracts import EpisodeResult, EvalReport
from orchestrator.stages import STAGE_BY_ID, stage_argv
from tests.orchestrator.test_pipeline import control, stages

SEL = {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}


def fake_eval_job(c, run_id, tmp_path, successes, p95=100.0, policy="lerobot:x/smolvla_panda_pick_v3"):
    """A finished eval stage job of this run whose output folder holds an EvalReport (policy None: no --policy)."""
    store = c.app.state.store
    out = tmp_path / "var" / "runs" / run_id / "eval" / f"job{successes}"
    out.mkdir(parents=True)
    eps = [EpisodeResult(episode=i, seed=5000 + i, object="vial", instruction="x", success=i < successes,
                         failed_stage=None if i < successes else "grasp", steps=50, latency_ms_p50=p95 / 2,
                         latency_ms_p95=p95, scene_variant="in_distribution") for i in range(40)]
    (out / "eval_report.json").write_text(EvalReport(model_version="m", dataset_versions=[], backend="fake",
                                                     episodes=eps).model_dump_json())
    job = store.create_job("stage", "eval", {}, run_id=run_id, worker="fake")
    argv = ["python", "-m", "engines.fake.cli", "eval", "--out", str(out)] + (["--policy", policy] if policy else [])
    store.update_job(job.id, state="done", result={"out": str(out), "argv": argv})


def test_train_stage_needs_a_train_worker_and_dispatches_the_recipe(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": SEL}).json()
    assert stages(c, run["id"])["train"]["reason"] == "no online worker with capability 'train'"

    c = control(tmp_path / "t", worker_capabilities=["convert", "train"])
    run = c.post("/api/runs", json={"selection": SEL}).json()
    st = stages(c, run["id"])["train"]
    assert st["enabled"] and st["worker"] == "fake"
    r = c.post(f"/api/runs/{run['id']}/stages/train", json={"params": {"dataset": "x/pick_genesis@v1"}})
    assert r.status_code == 400 and "repo-id" in r.json()["detail"]
    job = c.post(f"/api/runs/{run['id']}/stages/train", json={"params": {
        "model": "xvla", "dataset": "x/pick_genesis@v1", "repo-id": "x/xvla_panda_genesis_v2", "steps": 10}}).json()
    argv = job["result"]["argv"]
    assert argv[1:5] == ["-m", "train.recipe", "--run", "--out"] and argv[5] == job["result"]["out"]
    assert argv[6:] == ["--model", "xvla", "--dataset", "x/pick_genesis@v1", "--repo-id", "x/xvla_panda_genesis_v2",
                        "--steps", "10"]


def test_register_then_promote_from_the_run_eval(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": SEL}).json()
    st = stages(c, run["id"])
    assert st["register"]["reason"] == "needs 'eval' to finish first" and st["register"]["worker"] is None
    assert st["promote"]["reason"] == "needs 'register' to finish first"

    fake_eval_job(c, run["id"], tmp_path, successes=30)
    assert stages(c, run["id"])["register"]["enabled"]
    reg = c.post(f"/api/runs/{run['id']}/stages/register", json={"params": {
        "ckpt": "x/smolvla_panda_pick_v3", "datasets": "x/pick_genesis@v1, x/pick_robosuite@v1"}}).json()
    assert reg["state"] == "done" and reg["result"]["version"] == "v1"
    version = c.get("/api/registry").json()["versions"][0]
    assert version["card"]["ckpt_uri"] == "x/smolvla_panda_pick_v3"
    assert version["card"]["dataset_versions"] == ["x/pick_genesis@v1", "x/pick_robosuite@v1"]
    assert list(version["eval_reports"]) == ["fake"]

    pro = c.post(f"/api/runs/{run['id']}/stages/promote", json={"params": {}}).json()
    assert pro["state"] == "failed" and "no eval on robosuite" in pro["result"]["error"]
    assert c.get(f"/api/jobs/{pro['id']}/log").json()["text"].startswith("no eval on robosuite")


def test_local_stage_errors_become_failed_jobs(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": SEL}).json()
    fake_eval_job(c, run["id"], tmp_path, successes=1, policy="url:http://127.0.0.1:8720")
    reg = c.post(f"/api/runs/{run['id']}/stages/register", json={"params": {}}).json()
    assert reg["state"] == "failed" and "ckpt" in reg["result"]["error"]
    with_bad = stage_argv(STAGE_BY_ID["promote"], {"version": "v9"})
    assert with_bad == ["--version", "v9"]
    pro = c.post(f"/api/runs/{run['id']}/stages/promote", json={"params": {"version": "v9"}})
    assert pro.status_code == 409 and "register" in json.dumps(pro.json())


def test_register_checks_the_checkpoint_the_eval_ran(tmp_path):
    c = control(tmp_path)
    run = c.post("/api/runs", json={"selection": SEL}).json()
    fake_eval_job(c, run["id"], tmp_path, successes=30)
    bad = c.post(f"/api/runs/{run['id']}/stages/register", json={"params": {"ckpt": "x/other"}}).json()
    assert bad["state"] == "failed" and "x/smolvla_panda_pick_v3" in bad["result"]["error"]
    ok = c.post(f"/api/runs/{run['id']}/stages/register", json={"params": {}}).json()
    assert ok["state"] == "done"
    assert c.get("/api/registry").json()["versions"][0]["card"]["ckpt_uri"] == "x/smolvla_panda_pick_v3"


def test_register_refuses_an_eval_of_the_expert(tmp_path):
    for policy in ("expert", None):
        c = control(tmp_path / str(policy))
        run = c.post("/api/runs", json={"selection": SEL}).json()
        fake_eval_job(c, run["id"], tmp_path / str(policy), successes=30, policy=policy)
        reg = c.post(f"/api/runs/{run['id']}/stages/register", json={"params": {"ckpt": "x/m"}}).json()
        assert reg["state"] == "failed" and "expert" in reg["result"]["error"]
