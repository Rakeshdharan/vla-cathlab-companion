"""Train-grid coverage from meta/episodes_extra.jsonl (one JSON line per episode with its EpisodeConfig)."""
import json
from collections import Counter
from pathlib import Path
from dataengine.variation import VariationSpec, train_configs
from dataset_intel.model import Coverage


def grid_key(cfg: dict) -> str:
    return f"{cfg['object']}|{cfg['layout_id']}|{cfg['obj_color_id']}|{cfg['table_color_id']}"


def load_extra(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def coverage(extra: list[dict], spec: VariationSpec) -> Coverage:
    cells: Counter = Counter()
    filled: set[str] = set()
    for row in extra:
        cfg = row["config"]
        filled.add(grid_key(cfg))
        cells[f"{grid_key(cfg)}|{row.get('source_sim', '?')}"] += 1
    grid = sorted({grid_key(c.model_dump()) for c in train_configs(spec)})
    return Coverage(cells=dict(sorted(cells.items())), empty_cells=[g for g in grid if g not in filled],
                    train_cells=len(grid), per_object=dict(Counter(r["config"]["object"] for r in extra)),
                    per_source_sim=dict(Counter(r.get("source_sim", "?") for r in extra)))
