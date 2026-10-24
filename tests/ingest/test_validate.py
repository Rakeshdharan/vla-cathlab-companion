import json
from engines.registry import IO_PROFILES
from ingest.validate import contract_checks, main

STATS = {k: {s: [0.0] for s in ("mean", "std", "min", "max")} for k in ("observation.state", "action")}


def panda_info(action_dim=7, fps=20):
    return {"fps": fps, "total_tasks": 9, "features": {
        "action": {"shape": [action_dim]}, "observation.state": {"shape": [8]},
        "observation.images.agentview": {"shape": [256, 256, 3]}, "observation.images.eye_in_hand": {"shape": [256, 256, 3]}}}


def test_panda_contract_ok_and_failures():
    assert all(c.ok for c in contract_checks(panda_info(), STATS, IO_PROFILES["panda_libero"]))
    bad = {c.name for c in contract_checks(panda_info(action_dim=16, fps=30), {}, IO_PROFILES["panda_libero"]) if not c.ok}
    assert {"fps", "action_dim", "stats action", "stats observation.state"} <= bad


def test_fr3_profile_without_obs_shapes(tmp_path):
    info = {"fps": 30, "total_tasks": 1, "features": {"action": {"shape": [16]}, "observation.state": {"shape": [16]},
                                                      "observation.images.external_camera": {"shape": [224, 224, 3]}}}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    (tmp_path / "meta/stats.json").write_text(json.dumps(STATS))
    assert main([str(tmp_path), "--io", "fr3_duo_16d", "--no-intel", "--out", str(tmp_path / "out")]) == 0
    assert "Contract OK" in (tmp_path / "out/ingest.md").read_text()
