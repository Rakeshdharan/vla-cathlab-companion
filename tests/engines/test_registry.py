import pytest
from contracts.specs import SceneSpec
from engines.registry import IO_PROFILES, EngineError, get_engine, load_engines, load_manifest

MANIFEST = """id: fake
label: Fake engine
env: {kind: conda, name: fake}
worker: fake
embodiments: [panda]
io: panda_libero
variation: configs/variation/cathlab_v1.yaml
task: configs/tasks/cathlab_pick.yaml
capabilities:
  preview: ready
  expert: {analytic: ready, graspgen: planned}
  generate: in_progress
entry: engines.fake.cli
"""


def write(tmp_path, text=MANIFEST, folder="fake"):
    d = tmp_path / folder
    d.mkdir()
    (d / "engine.yaml").write_text(text)
    return d / "engine.yaml"


def test_repo_engines_load_and_robosuite_is_ready_for_data():
    rs = load_engines()["robosuite"]
    assert rs.worker == "robosuite" and rs.entry == "engines.robosuite.cli"
    assert rs.io_profile.action_dim == 7 and rs.io_profile.obs_shapes["state"] == (8,)
    for cap in ("preview", "expert.analytic", "generate", "convert"):
        assert rs.status(cap) == "ready", cap


def test_status_handles_groups_and_unknown(tmp_path):
    m = load_manifest(write(tmp_path))
    assert m.status("preview") == "ready"
    assert m.status("expert.analytic") == "ready" and m.status("expert.graspgen") == "planned"
    assert m.status("expert") == "ready"          # a group reports its best child
    assert m.status("generate") == "in_progress"
    assert m.status("rollout") == "planned" and m.status("expert.mimicgen") == "planned"


def test_id_must_match_folder(tmp_path):
    with pytest.raises(EngineError, match="folder"):
        load_manifest(write(tmp_path, folder="other"))


def test_unknown_io_profile_rejected(tmp_path):
    with pytest.raises(ValueError, match="io profile"):
        load_manifest(write(tmp_path, MANIFEST.replace("panda_libero", "nope")))


def test_get_engine_unknown(tmp_path):
    with pytest.raises(EngineError, match="no engine plugin 'mujoco'"):
        get_engine("mujoco", tmp_path)


def test_io_profiles_and_newton_backend_literal():
    assert IO_PROFILES["fr3_duo_16d"].action_dim == 16 and IO_PROFILES["fr3_duo_16d"].fps == 30
    assert SceneSpec(scene_id="s", backend="newton", asset="", objects={}, cameras=[]).backend == "newton"
