from fastapi.testclient import TestClient

from orchestrator.control import create_control_app
from orchestrator.videos import load_videos
from tests.orchestrator.test_pipeline import ROOT

VIDEOS = """videos:
  - {label: "SmolVLA v2 hands over the vial", engine: robosuite, object: vial, model: cathlab-pick@v2,
     success: true, url: "https://huggingface.co/x/resolve/main/robosuite_vial.mp4"}
  - {label: "local clip", path: "clips/demo.mp4"}
"""


def test_load_videos_missing_file_returns_empty(tmp_path):
    assert load_videos(tmp_path / "nope.yaml") == []


def test_video_entry_requires_exactly_one_of_url_or_path():
    from orchestrator.videos import VideoEntry
    import pytest

    with pytest.raises(ValueError):
        VideoEntry(label="x")
    with pytest.raises(ValueError):
        VideoEntry(label="x", url="https://a", path="b")
    with pytest.raises(ValueError):
        VideoEntry(label="x", url="not-http")
    VideoEntry(label="x", url="https://a")
    VideoEntry(label="x", path="a/b.mp4")


def test_api_videos_lists_recorded_and_local(tmp_path):
    cfg = tmp_path / "demo_videos.yaml"
    cfg.write_text(VIDEOS)
    var = tmp_path / "var"
    app = create_control_app(ROOT, var_dir=var, catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend",
                             videos_path=cfg)
    c = TestClient(app)

    body = c.get("/api/videos").json()
    assert len(body["recorded"]) == 2
    assert body["recorded"][0]["src"] == "https://huggingface.co/x/resolve/main/robosuite_vial.mp4"
    assert body["recorded"][0]["success"] is True
    assert body["recorded"][1]["src"] == "/api/videos/file?path=clips/demo.mp4"
    assert body["local"] == []

    # a done rollout job with a rollout.json + video is listed in "local"
    out = var / "runs" / "r1" / "rollout" / "j1"
    out.mkdir(parents=True)
    (out / "rollout.json").write_text(
        '{"success": true, "steps": 42, "failed_stage": null, "object": "vial", "policy": "expert", '
        '"instruction": "pick the vial", "video": "rollout.mp4"}')
    (out / "rollout.mp4").write_bytes(b"fake-mp4")
    job = app.state.store.create_job("stage", "rollout", {}, run_id="r1", worker="w")
    app.state.store.update_job(job.id, state="done", result={"out": str(out)})

    body = c.get("/api/videos").json()
    assert len(body["local"]) == 1
    entry = body["local"][0]
    assert entry["job_id"] == job.id and entry["run_id"] == "r1"
    assert entry["object"] == "vial" and entry["policy"] == "expert"
    assert entry["success"] is True and entry["steps"] == 42 and entry["failed_stage"] is None
    assert entry["video_url"] == f"/api/jobs/{job.id}/file?path=rollout.mp4"
    assert c.get(entry["video_url"]).status_code == 200


def test_videos_file_path_outside_var_is_rejected(tmp_path):
    var = tmp_path / "var"
    var.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    app = create_control_app(ROOT, var_dir=var, catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend")
    c = TestClient(app)
    assert c.get("/api/videos/file", params={"path": "../outside.mp4"}).status_code == 400
    assert c.get("/api/videos/file", params={"path": "nope.mp4"}).status_code == 404


def test_videos_file_rejects_non_media_suffixes(tmp_path):
    var = tmp_path / "var"
    var.mkdir()
    # Create test files with various suffixes
    (var / "test.db").write_bytes(b"database")
    (var / "config.yaml").write_bytes(b"config")
    (var / "demo.gif").write_bytes(b"image")
    app = create_control_app(ROOT, var_dir=var, catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend")
    c = TestClient(app)
    # .db and .yaml files under var/ should return 400
    assert c.get("/api/videos/file", params={"path": "test.db"}).status_code == 400
    assert c.get("/api/videos/file", params={"path": "config.yaml"}).status_code == 400
    # .gif (allowed media file) should return 200
    assert c.get("/api/videos/file", params={"path": "demo.gif"}).status_code == 200


def test_api_videos_skips_non_dict_rollout_json(tmp_path):
    cfg = tmp_path / "demo_videos.yaml"
    cfg.write_text(VIDEOS)
    var = tmp_path / "var"
    app = create_control_app(ROOT, var_dir=var, catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend",
                             videos_path=cfg)
    c = TestClient(app)

    # Create a done rollout job with invalid (non-dict) rollout.json
    out = var / "runs" / "r1" / "rollout" / "j1"
    out.mkdir(parents=True)
    (out / "rollout.json").write_text("[]")  # Array instead of dict
    job = app.state.store.create_job("stage", "rollout", {}, run_id="r1", worker="w")
    app.state.store.update_job(job.id, state="done", result={"out": str(out)})

    # Create a valid rollout job for comparison
    out2 = var / "runs" / "r2" / "rollout" / "j2"
    out2.mkdir(parents=True)
    (out2 / "rollout.json").write_text(
        '{"success": true, "steps": 10, "failed_stage": null, "object": "syringe", "policy": "assist", '
        '"instruction": "inject", "video": "rollout.mp4"}')
    (out2 / "rollout.mp4").write_bytes(b"fake-mp4")
    job2 = app.state.store.create_job("stage", "rollout", {}, run_id="r2", worker="w")
    app.state.store.update_job(job2.id, state="done", result={"out": str(out2)})

    # /api/videos should still return 200 and include only the valid job
    body = c.get("/api/videos").json()
    assert c.get("/api/videos").status_code == 200
    assert len(body["local"]) == 1
    assert body["local"][0]["job_id"] == job2.id


def test_repo_demo_videos_config_is_valid():
    videos = load_videos(ROOT / "configs/demo_videos.yaml")
    assert all(v.url and v.url.startswith("https://") and v.path is None for v in videos)
    assert any(v.success and v.engine == "robosuite" for v in videos)
