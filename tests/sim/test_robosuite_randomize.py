import os

os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("robosuite")

from dataengine.variation import VariationSpec, train_configs  # noqa: E402
from sim.robosuite.env import CathlabPickPlace  # noqa: E402
from sim.robosuite.randomize import apply_episode_config, kelvin_to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")


@pytest.fixture(scope="module")
def env():
    e = CathlabPickPlace(SPEC, use_camera_obs=False, has_offscreen_renderer=False)
    yield e
    e.close()


def _visual_rgba(env, name):
    m = env.sim.model
    return [m.geom_rgba[m.geom_name2id(g)].copy() for g in env.objects[name].visual_geoms]


def test_kelvin_to_rgb_is_warm_to_cool_and_normalised():
    warm, cool = kelvin_to_rgb(4000), kelvin_to_rgb(7500)
    assert warm.max() == pytest.approx(1.0) and cool.max() == pytest.approx(1.0)
    assert warm[0] > warm[2] and cool[2] > cool[0]


def test_apply_sets_colours_physics_and_lights_without_compounding(env):
    a, b = train_configs(SPEC)[0], train_configs(SPEC)[21]
    assert (a.obj_color_id, a.table_color_id) != (b.obj_color_id, b.table_color_id)
    m = env.sim.model
    for cfg in (a, b, b):
        env.reset()
        apply_episode_config(env, SPEC, cfg)
        for name, obj in SPEC.objects.items():
            for rgba in _visual_rgba(env, name):
                np.testing.assert_allclose(rgba, SPEC.obj_colors[cfg.obj_color_id][name])
            body = env.obj_body_ids[name]
            assert m.body_mass[body] == pytest.approx(env.nominal_inertial[name][0] * cfg.mass_scale)
            np.testing.assert_allclose(m.body_inertia[body], env.nominal_inertial[name][1] * cfg.mass_scale)
            for g in env.objects[name].contact_geoms:
                assert m.geom_friction[m.geom_name2id(g)][0] == pytest.approx(cfg.friction)
        table = m.geom_name2id("table_visual")
        assert m.geom_matid[table] == -1
        np.testing.assert_allclose(m.geom_rgba[table], SPEC.table_colors[cfg.table_color_id])
        light = env.nominal_lights["diffuse"][0] * cfg.light_intensity * kelvin_to_rgb(cfg.light_temp_k)
        np.testing.assert_allclose(m.light_diffuse[0], light, rtol=1e-6)
