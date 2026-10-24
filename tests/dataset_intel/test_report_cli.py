"""End to end on a tiny synthetic LeRobot v3 root (needs pyarrow)."""
import json
import numpy as np
import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402
from dataset_intel.report import main  # noqa: E402


def write_root(root, lengths):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"fps": 20, "robot_type": "panda"}))
    rng = np.random.default_rng(0)
    rows = {"episode_index": [], "frame_index": [], "observation.state": [], "action": []}
    meta = []
    t0 = 0.0
    for ep, n in enumerate(lengths):
        for f in range(n):
            a = list(rng.uniform(-0.5, 0.5, 7)); a[6] = 1.0 if f > n // 2 else -1.0
            rows["episode_index"].append(ep); rows["frame_index"].append(f)
            rows["observation.state"].append(list(rng.normal(0, 1, 8))); rows["action"].append(a)
        vid = n if ep else n - 3                                   # episode 0's video is 3 frames short
        meta.append({"episode_index": ep, "length": n, "videos/observation.images.agentview/from_timestamp": t0,
                     "videos/observation.images.agentview/to_timestamp": t0 + vid / 20})
        t0 += vid / 20
    pq.write_table(pa.table(rows), root / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(meta), root / "meta/episodes/chunk-000/file-000.parquet")


def test_cli_writes_report(tmp_path):
    write_root(tmp_path / "ds", [60, 62, 61, 59, 300])
    main([str(tmp_path / "ds"), "--out", str(tmp_path / "out")])
    r = json.loads((tmp_path / "out/report.json").read_text())
    assert r["dataset"]["episodes"] == 5 and r["dataset"]["frames"] == 542
    found = {(i["episode"], i["kind"]) for i in r["issues"]}
    assert (0, "frame_count_mismatch") in found and (4, "length_outlier") in found
    assert r["readiness"]["valid_fraction"] == 0.8 and r["coverage"] is None
