"""Genesis scene, I/O convention and randomisation (engine env only; one scene for the module).

    PYTHONPATH=. timeout 600 ~/venvs/genesis/bin/python -m pytest -q tests/sim/test_genesis_env.py
"""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("genesis")

from contracts import STAGES  # noqa: E402
from dataengine.geometry import quat_conj, quat_mul, quat_to_axis_angle  # noqa: E402
from dataengine.variation import VariationSpec, object_poses, train_configs  # noqa: E402
from sim.genesis.env import _to_np  # noqa: E402
from sim.genesis.expert_sim import GenesisExpertSim, state_axis_angle  # noqa: E402
from sim.genesis.randomize import (GENESIS_AXES, apply_episode_config, apply_genesis_axes, genesis_axes,  # noqa: E402
                                   kelvin_to_rgb)

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
# robosuite at HOME_QPOS (RobosuiteExpertSim, measured on 2026-10-23): state and grip-site pose (wxyz)
RS_STATE = np.array([0.45, 0.0, 0.40, 3.1352, 0.00077, 0.19983, 0.0387, -0.0387])
RS_EE = np.array([0.45, 0.0, 0.40, 0.04499, 0.7055, 0.70585, 0.04497])


@pytest.fixture(scope="module")
def sim():
    s = GenesisExpertSim(SPEC)
    yield s
    s.env.close()


def test_obs_shapes_and_flags(sim):
    obs = sim.reset(train_configs(SPEC)[0])
    assert obs["state"].shape == (8,) and obs["state"].dtype == np.float32
    for cam in ("agentview", "eye_in_hand"):
        assert obs[cam].shape == (256, 256, 3) and obs[cam].dtype == np.uint8 and obs[cam].std() > 5
    obs, info = sim.step(np.zeros(7, dtype=np.float32))
    assert set(info.stage_flags) == set(STAGES) and not info.success


def test_state_convention_at_home(sim):
    obs = sim.reset(train_configs(SPEC)[0])
    np.testing.assert_allclose(obs["state"][:3], RS_STATE[:3], atol=1e-3)          # TCP 0.097 m along hand +z
    np.testing.assert_allclose(obs["state"][3:6], RS_STATE[3:6], atol=2e-3)        # hand axis-angle
    assert obs["state"][6] == pytest.approx(0.04, abs=2e-3) and obs["state"][7] == pytest.approx(-0.04, abs=2e-3)
    ee = sim.get_ee_pose()
    np.testing.assert_allclose(ee[:3], RS_EE[:3], atol=1e-3)
    assert np.linalg.norm(quat_to_axis_angle(quat_mul(ee[3:], quat_conj(RS_EE[3:])))) < 2e-3


def test_actions_move_in_the_base_frame(sim):
    sim.reset(train_configs(SPEC)[0])
    e0 = sim.get_ee_pose()
    for _ in range(3):
        sim.step(np.array([1, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    e1 = sim.get_ee_pose()
    d = e1[:3] - e0[:3]
    assert d[0] > 0.02 and abs(d[1]) < 2e-3 and abs(d[2]) < 5e-3
    for _ in range(3):
        sim.step(np.array([0, 0, 0, 0, 0, 1, -1], dtype=np.float32))
    rot = quat_to_axis_angle(quat_mul(sim.get_ee_pose()[3:], quat_conj(e1[3:])))
    assert rot[2] > 0.05 and np.linalg.norm(rot[:2]) < 0.3 * rot[2]


def test_objects_rest_on_the_table(sim):
    cfg = train_configs(SPEC)[0]
    sim.reset(cfg)
    before = {n: sim.get_object_pose(n) for n in SPEC.objects}
    sim.env.settle(40)
    for name, o in SPEC.objects.items():
        p = sim.get_object_pose(name)
        assert np.linalg.norm(p[:3] - before[name][:3]) < 2e-3, name
        assert p[2] - o.center_z() > -1e-3, name                                    # no penetration
        np.testing.assert_allclose(p[:2], object_poses(SPEC, cfg)[name][:2], atol=2e-3)


def _snapshot(env) -> dict:
    ctx = env.scene.visualizer.context
    out = {}
    for name, obj in [*env.objects.items(), ("table", env.table)]:
        node = ctx.rigid_nodes[obj.vgeoms[0].uid]
        out[name + ".rgba"] = node.mesh.primitives[0].material.baseColorFactor.tolist()
        if name != "table":
            out[name + ".mass"] = float(_to_np(obj.get_mass()))
            out[name + ".friction"] = [float(_to_np(g.get_friction())) for g in obj.geoms]
    out["lights"] = [(n.light.color.tolist(), float(n.light.intensity)) for n in ctx._scene.light_nodes]
    out["ambient"] = np.asarray(ctx._scene.ambient_light).tolist()
    return out


def test_randomize_is_absolute_not_compounding(sim):
    env = sim.env
    a = train_configs(SPEC)[0]
    b = next(c for c in train_configs(SPEC) if c.obj_color_id != a.obj_color_id
             and c.table_color_id != a.table_color_id and c.mass_scale > a.mass_scale)
    snaps = []
    for cfg in (a, b, a):
        env.reset()
        apply_episode_config(env, SPEC, cfg)
        snaps.append(_snapshot(env))
    assert snaps[0] == snaps[2] and snaps[0] != snaps[1]
    for name, o in SPEC.objects.items():
        assert snaps[1][name + ".mass"] == pytest.approx(o.mass * b.mass_scale, rel=1e-4)
        assert snaps[1][name + ".friction"] == pytest.approx([b.friction] * len(snaps[1][name + ".friction"]))
        assert snaps[1][name + ".rgba"] == pytest.approx(SPEC.obj_colors[b.obj_color_id][name])
    assert snaps[1]["table.rgba"] == pytest.approx(SPEC.table_colors[b.table_color_id])
    assert snaps[1]["lights"][0][1] == pytest.approx(snaps[0]["lights"][0][1] * b.light_intensity / a.light_intensity)


def _view_snapshot(env) -> dict:
    ctx = env.scene.visualizer.context
    av = env.cameras["agentview"]
    return {"cam_pos": np.asarray(_to_np(av.pos)).round(6).tolist(),
            "cam_lookat": np.asarray(_to_np(av.lookat)).round(6).tolist(),
            "light_dirs": [(-np.asarray(n.matrix)[:3, 2]).round(6).tolist() for n in ctx._scene.light_nodes
                           if type(n.light).__name__ == "DirectionalLight"]}


def test_genesis_axes_are_seeded_bounded_and_absolute(sim):
    env = sim.env
    cfgs = train_configs(SPEC)
    a, b = cfgs[0], cfgs[1]
    assert genesis_axes(a) == genesis_axes(a) and genesis_axes(a) != genesis_axes(b)
    for cfg in cfgs[:50]:
        ax = genesis_axes(cfg)
        assert np.abs(ax.cam_pos_offset).max() <= GENESIS_AXES.cam_pos and \
            np.abs(ax.cam_lookat_offset).max() <= GENESIS_AXES.cam_lookat
        assert GENESIS_AXES.light_elevation_deg[0] <= ax.light_elevation_deg <= GENESIS_AXES.light_elevation_deg[1]
    nominal = _view_snapshot(env)
    snaps = []
    for cfg in (a, b, a):
        apply_genesis_axes(env, genesis_axes(cfg))
        snaps.append(_view_snapshot(env))
    assert snaps[0] == snaps[2] and snaps[0] != snaps[1] and snaps[0] != nominal
    np.testing.assert_allclose(np.array(snaps[1]["cam_pos"]) - np.array(nominal["cam_pos"]),
                               genesis_axes(b).cam_pos_offset, atol=1e-5)
    apply_genesis_axes(env, None)
    assert _view_snapshot(env) == nominal


def test_expert_sim_applies_genesis_axes_only_when_enabled(sim):
    cfg = train_configs(SPEC)[3]
    sim.genesis_axes = False
    sim.reset(cfg)
    nominal = _view_snapshot(sim.env)
    sim.genesis_axes = True
    obs = sim.reset(cfg)
    assert sim.axes == genesis_axes(cfg) and _view_snapshot(sim.env) != nominal
    assert obs["agentview"].std() > 5
    sim.genesis_axes = False
    sim.reset(cfg)
    assert sim.axes is None and _view_snapshot(sim.env) == nominal


def test_kelvin_to_rgb_is_warm_to_cool_and_normalised():
    warm, cool = kelvin_to_rgb(4000), kelvin_to_rgb(7500)
    assert warm.max() == pytest.approx(1.0) and cool.max() == pytest.approx(1.0)
    assert warm[0] > warm[2] and cool[2] > cool[0]


def test_state_axis_angle_tie_break_at_pi():
    # The Menagerie hand at HOME_QPOS is rotated exactly pi (w = 0); the sign of w is float noise (+0 on GPU, -eps on
    # CPU) and must not flip the state's axis-angle away from robosuite's (+3.135, 0, +0.2).
    v = np.array([0.99797, 0.0, 0.06361])
    for w in (0.0, -1e-9, 1e-9):
        for sign in (1, -1):
            aa = state_axis_angle(sign * np.array([w, *v]))
            np.testing.assert_allclose(aa, np.pi * v / np.linalg.norm(v), atol=1e-6)
    q = np.array([0.3, -0.5, 0.4, 0.707])                       # away from the cusp: plain quat_to_axis_angle
    np.testing.assert_allclose(state_axis_angle(q), quat_to_axis_angle(q))
    np.testing.assert_allclose(state_axis_angle(-q), quat_to_axis_angle(q))


def test_backend_from_env(monkeypatch):
    from sim.genesis.env import BACKEND_ENV, genesis_backend
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    assert genesis_backend() == "gpu"
    monkeypatch.setenv(BACKEND_ENV, "CPU")
    assert genesis_backend() == "cpu"
    monkeypatch.setenv(BACKEND_ENV, "metal")
    with pytest.raises(ValueError):
        genesis_backend()
