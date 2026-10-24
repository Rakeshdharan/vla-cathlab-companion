from fastapi.testclient import TestClient

from contracts import DatasetManifest, save_yaml
from orchestrator.control import create_control_app
from orchestrator.datasets import load_datasets
from tests.orchestrator.test_pipeline import ROOT

DATASETS = """datasets:
  - {repo_id: Rakeshdharan56/pick_robosuite, revision: v1, embodiment: panda, io: panda_libero, engine: robosuite,
     physics: MuJoCo 3.3.7, episodes: 384, frames: 69191, fps: 20, visibility: public}
  - {repo_id: Rakeshdharan56/pick_panda_v2, embodiment: panda, io: panda_libero, engine: merged,
     episodes: 1136, frames: 181025, fps: 20, visibility: local, local_path: data/pick_panda_v2}
"""


def test_repo_datasets_config_is_valid_and_pinned():
    ds = load_datasets(ROOT / "configs/datasets.yaml").datasets
    assert {"Rakeshdharan56/pick_robosuite", "Rakeshdharan56/pick_maniskill", "Rakeshdharan56/pick_genesis"} <= \
        {d.repo_id for d in ds}
    assert all(d.revision or d.visibility == "local" for d in ds)


def test_api_datasets_lists_published_and_local_manifests(tmp_path):
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(DATASETS)
    var = tmp_path / "var"
    shard = var / "runs" / "r1" / "generate" / "j1" / "raw" / "shard_000-of-001"
    shard.mkdir(parents=True)
    save_yaml(DatasetManifest(dataset_id="cathlab/pick_genesis", version="v1", source="scripted", task_id="pick",
                              scene_id="cathlab_v1_genesis", n_episodes=3, created_by="me"), shard / "manifest.yaml")
    app = create_control_app(ROOT, var_dir=var, catalog_path=ROOT / "configs/catalog.yaml",
                             workers_path=tmp_path / "none.yaml", frontend_dir=ROOT / "frontend",
                             datasets_path=cfg)
    body = TestClient(app).get("/api/datasets").json()
    assert [d["repo_id"] for d in body["published"]] == ["Rakeshdharan56/pick_robosuite", "Rakeshdharan56/pick_panda_v2"]
    assert body["published"][0]["url"] == "https://huggingface.co/datasets/Rakeshdharan56/pick_robosuite"
    assert body["published"][1]["url"] is None and body["published"][1]["local_path"] == "data/pick_panda_v2"
    (local,) = body["local"]
    assert local["dataset_id"] == "cathlab/pick_genesis" and local["n_episodes"] == 3
    assert local["path"] == "runs/r1/generate/j1/raw/shard_000-of-001" and local["run_id"] == "r1"


def test_data_tab_is_wired():
    html = (ROOT / "frontend/index.html").read_text()
    assert 'id="datasets-published"' in html and 'id="datasets-local"' in html
    assert "initData" in (ROOT / "frontend/js/app.js").read_text()
