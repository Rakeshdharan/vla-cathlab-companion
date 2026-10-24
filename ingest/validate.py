"""Contract check + quality report for a (provided or generated) LeRobot v3 dataset.

    uv run --with pyarrow python -m ingest.validate data/fr3_duo_handover_84ep --io fr3_duo_16d --out var/ingest/fr3_84ep

Checks meta/info.json against engines.registry.IO_PROFILES (action dim, fps, state/camera shapes) and meta/stats.json
(mean/std/min/max for state and action, needed for normalisation), then runs dataset_intel. Writes ingest.json/.md.
"""
import argparse
import json
from pathlib import Path
from pydantic import BaseModel, Field
from engines.registry import IO_PROFILES, IOProfile

STATS_KEYS = ("mean", "std", "min", "max")


class Check(BaseModel):
    name: str
    ok: bool
    detail: str = ""


class IngestReport(BaseModel):
    root: str
    io: str
    checks: list[Check] = Field(default_factory=list)
    readiness: float | None = None
    issues: int | None = None

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)


def _shape(features: dict, key: str) -> tuple | None:
    f = features.get(key)
    return tuple(f["shape"]) if f else None


def contract_checks(info: dict, stats: dict, profile: IOProfile) -> list[Check]:
    f = info.get("features", {})
    checks = [Check(name="fps", ok=int(info.get("fps", -1)) == profile.fps, detail=f"{info.get('fps')} vs {profile.fps}"),
              Check(name="action_dim", ok=_shape(f, "action") == (profile.action_dim,),
                    detail=f"{_shape(f, 'action')} vs ({profile.action_dim},)")]
    for key, shape in profile.obs_shapes.items():
        name = "observation.state" if key == "state" else f"observation.images.{key}"
        got = _shape(f, name)
        checks.append(Check(name=name, ok=got == tuple(shape), detail=f"{got} vs {tuple(shape)}"))
    if not profile.obs_shapes:                               # profile without obs shapes: at least state + 1 camera
        cams = [k for k in f if k.startswith("observation.images.")]
        checks.append(Check(name="observation.state", ok="observation.state" in f, detail=str(_shape(f, "observation.state"))))
        checks.append(Check(name="cameras", ok=bool(cams), detail=", ".join(cams)))
    for key in ("observation.state", "action"):
        missing = [s for s in STATS_KEYS if s not in stats.get(key, {})]
        checks.append(Check(name=f"stats {key}", ok=not missing, detail=f"missing {missing}" if missing else "mean/std/min/max"))
    tasks = info.get("total_tasks", 0)
    checks.append(Check(name="tasks", ok=tasks > 0, detail=f"{tasks} language instructions"))
    return checks


def render_md(r: IngestReport) -> str:
    lines = [f"# Ingest: `{r.root}` ({r.io})", "", f"**Contract {'OK' if r.ok else 'FAILED'}**"
             + (f" · dataset_intel readiness {r.readiness:.2f}, {r.issues} issues" if r.readiness is not None else ""),
             "", "| Check | OK | Detail |", "|---|---|---|"]
    lines += [f"| {c.name} | {'✅' if c.ok else '❌'} | {c.detail} |" for c in r.checks]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--io", required=True, choices=sorted(IO_PROFILES))
    ap.add_argument("--extra", help="episodes_extra.jsonl for dataset_intel coverage")
    ap.add_argument("--no-intel", action="store_true", help="contract checks only (no pyarrow needed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    info = json.loads((root / "meta/info.json").read_text())
    stats_path = root / "meta/stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    report = IngestReport(root=str(root), io=a.io, checks=contract_checks(info, stats, IO_PROFILES[a.io]))
    if not a.no_intel:
        from dataset_intel.report import main as intel_main
        intel_main([str(root), "--io", a.io, "--out", str(out / "intel")] + (["--extra", a.extra] if a.extra else []))
        intel = json.loads((out / "intel/report.json").read_text())
        report.readiness, report.issues = intel["readiness"]["score"], len(intel["issues"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "ingest.json").write_text(report.model_dump_json(indent=2))
    (out / "ingest.md").write_text(render_md(report))
    print(f"contract {'OK' if report.ok else 'FAILED'} -> {out}/ingest.md")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
