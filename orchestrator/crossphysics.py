"""Cross-physics table for the REPORT tab: reports/robosuite/cross_physics_v1.json (train/cross_physics_report.py)
reshaped into rows (model, trained on) x engine columns."""
import json
from pathlib import Path

ENGINE_ORDER = ("robosuite", "maniskill", "genesis")
TABLE = "reports/robosuite/cross_physics_v1.json"


def load_cross_physics(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        return {"source": None, "engines": [], "rows": []}
    entries = json.loads(p.read_text()).values()
    engines = sorted({e["engine"] for e in entries},
                     key=lambda e: (ENGINE_ORDER.index(e) if e in ENGINE_ORDER else len(ENGINE_ORDER), e))
    rows: dict[str, dict] = {}
    for e in entries:
        trained = e.get("trained_on")
        trained = None if trained in (None, "—", "") else trained
        row = rows.setdefault(e["model"], {"model": e["model"], "trained_on": trained, "cells": {}})
        o = e["overall"]
        row["cells"][e["engine"]] = {"rate": o["success_rate"], "ci_low": o["ci_low"], "ci_high": o["ci_high"],
                                     "n": o["n"], "in_domain": trained == e["engine"]}
    def order(r: dict) -> tuple:
        t = r["trained_on"]
        return (t is not None, ENGINE_ORDER.index(t) if t in ENGINE_ORDER else len(ENGINE_ORDER), r["model"])

    ordered = sorted(rows.values(), key=order)
    return {"source": TABLE, "engines": engines, "rows": ordered}
