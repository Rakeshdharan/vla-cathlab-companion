from pathlib import Path
import numpy as np
import pytest
from dataengine.variation import (TIERS, VariationSpec, eval_configs, generation_order, layouts, object_poses,
                                  train_configs)

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


def test_spec_loads_three_objects_and_five_palettes():
    assert list(SPEC.objects) == ["bottle", "vial", "syringe"]
    assert len(SPEC.obj_colors) == 5 and len(SPEC.table_colors) == 5


def test_layouts_are_deterministic_separated_and_in_region():
    a, b = layouts(SPEC), layouts(SPEC)
    assert a == b and len(a) == 10
    cx, cy = SPEC.scene.pick_region_center
    sx, sy = SPEC.scene.pick_region_size
    for lay in a:
        pts = np.array([p[:2] for p in lay.values()])
        assert np.all(np.abs(pts[:, 0] - cx) <= sx / 2) and np.all(np.abs(pts[:, 1] - cy) <= sy / 2)
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1)[np.triu_indices(3, 1)]
        assert d.min() >= SPEC.scene.min_separation


def test_train_grid_is_384_and_uses_only_train_values():
    cfgs = train_configs(SPEC)
    assert len(cfgs) == 3 * 8 * 4 * 4 == 384
    assert {c.layout_id for c in cfgs} == set(SPEC.layout_split.train)
    assert {c.obj_color_id for c in cfgs} == set(SPEC.obj_color_split.train)
    assert {c.table_color_id for c in cfgs} == set(SPEC.table_color_split.train)
    assert all(0.7 <= c.mass_scale <= 1.3 and 0.5 <= c.friction <= 1.0 for c in cfgs)
    assert train_configs(SPEC) == cfgs


def test_eval_tiers_hold_out_the_right_axes():
    seeds = [1000, 1001, 1002, 1003]
    allh = eval_configs(SPEC, "all_held_out", seeds)
    assert len(allh) == 12
    assert all(c.layout_id in (8, 9) and c.obj_color_id == 4 and c.table_color_id == 4 for c in allh)
    ind = eval_configs(SPEC, "in_distribution", seeds)
    assert all(c.layout_id < 8 and c.obj_color_id < 4 and c.table_color_id < 4 for c in ind)
    lay = eval_configs(SPEC, "held_out_layout", seeds)
    assert all(c.layout_id in (8, 9) and c.obj_color_id < 4 for c in lay)
    assert len(TIERS) == 5


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        eval_configs(SPEC, "nope", [1])


def test_object_poses_rest_on_table():
    poses = object_poses(SPEC, train_configs(SPEC)[0])
    assert poses["bottle"][2] == pytest.approx(0.125)
    assert poses["syringe"][2] == pytest.approx(0.0075)
    assert all(abs(np.linalg.norm(p[3:]) - 1) < 1e-9 for p in poses.values())


def test_generation_order_is_balanced_diverse_and_a_permutation():
    cfgs = train_configs(SPEC)
    ordered = generation_order(SPEC, cfgs)
    assert sorted(c.seed for c in ordered) == sorted(c.seed for c in cfgs)
    assert [c.object for c in ordered[:6]] == ["bottle", "vial", "syringe"] * 2
    first = ordered[:24]
    assert len({c.layout_id for c in first}) >= 6
    assert len({c.obj_color_id for c in first}) == 4 and len({c.table_color_id for c in first}) == 4
    assert generation_order(SPEC, cfgs) == ordered
