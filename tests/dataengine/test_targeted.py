import json
import pytest
from dataengine.variation import (TARGETED_SEED0, VariationSpec, cell_configs, generation_order, select_configs,
                                  train_configs)

SPEC = VariationSpec.load("configs/variation/cathlab_v1.yaml")
CELLS = [{"object": "bottle", "layout_id": 0, "obj_color_id": 0, "table_color_id": 0, "k": 3},
         {"object": "vial", "layout_id": 2, "obj_color_id": 2, "table_color_id": 2, "k": 2}]


def test_cell_configs_fix_the_cell_and_vary_the_rest():
    cfgs = cell_configs(SPEC, CELLS)
    assert len(cfgs) == 5 and all(c.tier == "targeted" for c in cfgs)
    bottles = [c for c in cfgs if c.object == "bottle"]
    assert {(c.layout_id, c.obj_color_id, c.table_color_id) for c in bottles} == {(0, 0, 0)}
    assert [c.seed for c in bottles] == [TARGETED_SEED0, TARGETED_SEED0 + 1, TARGETED_SEED0 + 2]
    assert len({c.mass_scale for c in bottles}) == 3
    assert cell_configs(SPEC, CELLS) == cfgs                                  # deterministic


@pytest.mark.parametrize("bad", [{"object": "scalpel"}, {"layout_id": 99}, {"obj_color_id": -1}, {"table_color_id": 50}])
def test_cell_configs_validate(bad):
    with pytest.raises(ValueError):
        cell_configs(SPEC, [{**CELLS[0], **bad}])


def test_select_configs_cells_split_and_shards(tmp_path):
    p = tmp_path / "next_run.json"
    p.write_text(json.dumps({"spec": "x", "cells": CELLS}))
    full = select_configs(SPEC, f"cells:{p}", 0, 5, 0, 1)
    assert full == generation_order(SPEC, cell_configs(SPEC, CELLS))
    assert [c.object for c in full[:2]] == ["bottle", "vial"]                 # objects interleaved
    shards = [select_configs(SPEC, f"cells:{p}", 0, 5, i, 2) for i in range(2)]
    assert sorted(c.seed for s in shards for c in s) == sorted(c.seed for c in full)
    assert select_configs(SPEC, f"cells:{p}", 0, 2, 0, 1) == full[:2]


def test_select_configs_train_matches_old_window_and_rejects_unknown():
    assert select_configs(SPEC, "train", 10, 6, 1, 2) == generation_order(SPEC, train_configs(SPEC))[10:16][1::2]
    with pytest.raises(ValueError, match="split"):
        select_configs(SPEC, "bogus", 0, 1, 0, 1)


def test_engines_share_the_function():
    import importlib
    from dataengine import variation
    for mod in ("engines.maniskill.stages", "engines.genesis.generate"):
        assert importlib.import_module(mod).select_configs is variation.select_configs
