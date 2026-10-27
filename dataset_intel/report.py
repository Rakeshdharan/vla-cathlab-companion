"""Dataset intelligence report for a LeRobot v3 dataset: validators, train-grid coverage and readiness (S5).

    python -m dataset_intel.report data/pick_panda_v2 --extra data/pick_panda_v2/meta/episodes_extra.jsonl \
        --io panda_libero --out var/intel/pick_panda_v2

Needs pyarrow (`uv run --with pyarrow ...` or the lerobot env). Writes <out>/report.json + <out>/report.md.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from dataset_intel import validators as V
from dataset_intel.coverage import coverage, load_extra
from dataset_intel.model import DatasetInfo, IntelReport, Readiness

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = ROOT / "configs/intel.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_dataset(root: str | Path) -> tuple[dict, dict[int, dict], dict[int, dict]]:
    """(info.json, episode -> {state, action} arrays, episode -> meta row) from a LeRobot v3 root."""
    import pyarrow.parquet as pq
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    data = [pq.read_table(p, columns=["episode_index", "frame_index", "observation.state", "action"]).to_pydict()
            for p in sorted((root / "data").glob("*/*.parquet"))]
    rows: dict[int, list] = {}
    for t in data:
        for ep, fi, s, a in zip(t["episode_index"], t["frame_index"], t["observation.state"], t["action"], strict=True):
            rows.setdefault(int(ep), []).append((fi, s, a))
    episodes = {}
    for ep, r in rows.items():
        r.sort(key=lambda x: x[0])
        episodes[ep] = {"state": np.asarray([x[1] for x in r], dtype=np.float64),
                        "action": np.asarray([x[2] for x in r], dtype=np.float64)}
    meta = {}
    for p in sorted((root / "meta/episodes").glob("*/*.parquet")):
        for row in pq.read_table(p).to_pylist():
            meta[int(row["episode_index"])] = row
    return info, episodes, meta


def video_frames(meta_row: dict, fps: int) -> dict[str, int]:
    out = {}
    for k in meta_row:
        if k.startswith("videos/") and k.endswith("/from_timestamp"):
            key = k[len("videos/"):-len("/from_timestamp")]
            out[key] = int(round((meta_row[f"videos/{key}/to_timestamp"] - meta_row[k]) * fps))
    return out


def build_report(info: dict, episodes: dict[int, dict], meta: dict[int, dict], io: str, cfg: dict,
                 extra: list[dict] | None = None, spec=None, root: str = "") -> IntelReport:
    fps = int(info["fps"])
    issues, idle = [], []
    for ep, d in sorted(episodes.items()):
        s, a = d["state"], d["action"]
        issues += V.check_nan(ep, s, a)
        issues += V.check_frozen(ep, s, fps, cfg["frozen_seconds"], cfg["frozen_eps"])
        issues += V.check_action_jump(ep, a, io, cfg["action_jump"][io])
        issues += V.check_gripper(ep, a, io)
        frac = V.idle_fraction(a, io)
        idle.append(frac)
        issues += V.check_idle(ep, frac, cfg["idle_fraction_max"])
        if ep in meta:
            issues += V.check_frame_counts(ep, len(a), video_frames(meta[ep], fps))
    issues += V.length_outliers({ep: len(d["action"]) for ep, d in episodes.items()}, cfg["length_iqr_k"])
    issues.sort(key=lambda i: (i.episode, i.kind))

    n = len(episodes)
    errored = {i.episode for i in issues if i.severity == "error"}
    cov = coverage(extra, spec) if extra and spec is not None else None
    comps = {"valid": 1 - len(errored) / max(n, 1), "idle": 1 - float(np.mean(idle)) if idle else 1.0}
    if cov is not None:
        comps["coverage"] = 1 - len(cov.empty_cells) / max(cov.train_cells, 1)
        counts = list(cov.per_object.values())
        comps["balance"] = min(counts) / max(counts) if counts else 0.0
    w = cfg["weights"]
    score = sum(w[k] * v for k, v in comps.items()) / sum(w[k] for k in comps)
    readiness = Readiness(score=round(score, 4), valid_fraction=comps["valid"],
                          coverage_fraction=comps.get("coverage"), object_balance=comps.get("balance"),
                          idle_penalty=1 - comps["idle"])
    dataset = DatasetInfo(root=str(root), robot_type=info.get("robot_type"), fps=fps, episodes=n,
                          frames=sum(len(d["action"]) for d in episodes.values()))
    return IntelReport(dataset=dataset, issues=issues, coverage=cov, readiness=readiness)


def render_md(r: IntelReport) -> str:
    d, rd = r.dataset, r.readiness
    kinds: dict[str, list[int]] = {}
    for i in r.issues:
        kinds.setdefault(f"{i.kind} ({i.severity})", []).append(i.episode)
    lines = [f"# Dataset intelligence: `{d.root}`", "",
             f"{d.episodes} episodes, {d.frames} frames, {d.fps} fps, robot_type `{d.robot_type}`.", "",
             f"**Readiness {rd.score:.2f}**: valid {rd.valid_fraction:.0%}"
             + (f", coverage {rd.coverage_fraction:.0%}" if rd.coverage_fraction is not None else "")
             + (f", object balance {rd.object_balance:.2f}" if rd.object_balance is not None else "")
             + f", idle {rd.idle_penalty:.1%}.", "", "## Issues", ""]
    if kinds:
        lines += ["| Kind | Episodes | Examples |", "|---|---|---|"]
        lines += [f"| {k} | {len(eps)} | {', '.join(map(str, eps[:8]))}{' …' if len(eps) > 8 else ''} |"
                  for k, eps in sorted(kinds.items())]
    else:
        lines.append("None.")
    if r.coverage is not None:
        c = r.coverage
        lines += ["", "## Coverage", "",
                  f"{c.train_cells - len(c.empty_cells)}/{c.train_cells} train-grid cells filled. "
                  f"Per object: {c.per_object}. Per source: {c.per_source_sim}."]
        if c.empty_cells:
            lines.append(f"Empty (object|layout|obj_color|table_color): {', '.join(c.empty_cells[:20])}"
                         + (" …" if len(c.empty_cells) > 20 else ""))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="LeRobot v3 dataset root (meta/, data/, videos/)")
    ap.add_argument("--extra", help="meta/episodes_extra.jsonl (enables coverage)")
    ap.add_argument("--spec", default=str(ROOT / "configs/variation/cathlab_v1.yaml"))
    ap.add_argument("--io", default="panda_libero", choices=sorted(V.GRIPPER))
    ap.add_argument("--config", default=str(ROOT / "configs/intel.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from dataengine.variation import VariationSpec
    info, episodes, meta = load_dataset(a.root)
    extra = load_extra(a.extra) if a.extra else None
    report = build_report(info, episodes, meta, a.io, load_config(a.config), extra,
                          VariationSpec.load(a.spec) if extra else None, root=a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.model_dump_json(indent=2))
    (out / "report.md").write_text(render_md(report))
    print(f"readiness {report.readiness.score:.2f}, {len(report.issues)} issues -> {out}/report.md")


if __name__ == "__main__":
    main()
