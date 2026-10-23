"""Apply an EpisodeConfig to CathlabGenesisEnv: poses, object/table colours, mass, friction and lighting.

Colours and lights are changed in the rasterizer after build (pyrender materials and light nodes), so the scene is
never rebuilt. Objects are built with white vertex colours and tinted through their material's base colour.

Genesis-only axes (`genesis_axes`, `apply_genesis_axes`) add agentview camera jitter and key-light direction on top."""
from dataclasses import dataclass

import numpy as np
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses


def kelvin_to_rgb(kelvin: float) -> np.ndarray:
    """Blackbody colour (Tanner Helland fit), scaled so the brightest channel is 1."""
    t = kelvin / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * np.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307)
    rgb = np.clip([r, g, b], 0.0, 255.0)
    return rgb / rgb.max()


def set_entity_color(env, entity, rgba) -> None:
    ctx = env.scene.visualizer.context
    for geom in entity.vgeoms:
        node = ctx.rigid_nodes.get(geom.uid)
        if node is None:
            continue
        for prim in node.mesh.primitives:
            prim.material.baseColorFactor = np.asarray(rgba, dtype=np.float32)
    ctx._scene._meshes_updated = True     # re-reads the materials at the next render


def set_lighting(env, intensity: float, kelvin: float) -> None:
    """Light intensity × Kelvin colour on the directional light and the ambient term, from the nominal values."""
    ctx = env.scene.visualizer.context
    tint = kelvin_to_rgb(kelvin)
    nominal = env.nominal_light
    for node in ctx._scene.light_nodes:
        node.light.color = nominal["color"] * tint
        node.light.intensity = nominal["intensity"] * intensity
    ctx._scene.ambient_light = np.clip(nominal["ambient"] * intensity * tint, 0.0, 1.0)
    ctx._scene.bump_revision()


def apply_episode_config(env, spec: VariationSpec, cfg: EpisodeConfig) -> None:
    """Call right after env.reset(), then settle. Every value is set from the spec's nominal values (absolute mass,
    friction, colours, lights), never scaled in place, so nothing compounds across episodes."""
    palette = spec.obj_colors[cfg.obj_color_id]
    for name, obj in env.objects.items():
        obj.set_mass(spec.objects[name].mass * cfg.mass_scale)
        obj.set_friction(cfg.friction)
        if env.render_enabled:
            set_entity_color(env, obj, palette[name])
    if env.render_enabled:
        set_entity_color(env, env.table, spec.table_colors[cfg.table_color_id])
        set_lighting(env, cfg.light_intensity, cfg.light_temp_k)
    env.set_object_poses(object_poses(spec, cfg))


@dataclass(frozen=True)
class GenesisAxesRanges:
    """Genesis-only nuisance axes on top of cathlab_v1 (the frozen grid is untouched, so episodes stay paired with
    robosuite/ManiSkill): agentview camera jitter and key-light direction."""
    cam_pos: float = 0.03                          # m, uniform per axis on the agentview eye
    cam_lookat: float = 0.03                       # m, uniform per axis on the agentview look-at point
    light_elevation_deg: tuple[float, float] = (50.0, 85.0)   # below the horizon (nominal ~70)
    light_azimuth_deg: tuple[float, float] = (-180.0, 180.0)


GENESIS_AXES = GenesisAxesRanges()


@dataclass(frozen=True)
class GenesisAxes:
    cam_pos_offset: tuple[float, float, float]
    cam_lookat_offset: tuple[float, float, float]
    light_elevation_deg: float
    light_azimuth_deg: float

    def light_dir(self) -> np.ndarray:
        el, az = np.deg2rad(self.light_elevation_deg), np.deg2rad(self.light_azimuth_deg)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), -np.sin(el)])


def genesis_axes(cfg: EpisodeConfig, ranges: GenesisAxesRanges = GENESIS_AXES) -> GenesisAxes:
    """Deterministic from cfg.seed, independent of the cathlab_v1 draws (own RNG stream)."""
    rng = np.random.default_rng([cfg.seed, 0x6E5E5])
    r = lambda lim: tuple(float(v) for v in rng.uniform(-lim, lim, 3))  # noqa: E731
    return GenesisAxes(cam_pos_offset=r(ranges.cam_pos), cam_lookat_offset=r(ranges.cam_lookat),
                       light_elevation_deg=float(rng.uniform(*ranges.light_elevation_deg)),
                       light_azimuth_deg=float(rng.uniform(*ranges.light_azimuth_deg)))


def apply_genesis_axes(env, axes: GenesisAxes | None) -> None:
    """Set the agentview pose and the key-light direction from nominal + axes (None = nominal). Absolute, never
    compounding. No-op without rendering."""
    if not env.render_enabled:
        return
    from genesis.utils.geom import z_up_to_R
    from sim.genesis.env import AGENTVIEW, CORE_ORIGIN, look_at
    av = look_at(CORE_ORIGIN + AGENTVIEW["pos"], AGENTVIEW["dir"], AGENTVIEW["up"])
    light_dir = env.nominal_light["dir"]
    if axes is not None:
        av["pos"] = av["pos"] + axes.cam_pos_offset
        av["lookat"] = av["lookat"] + axes.cam_lookat_offset
        light_dir = axes.light_dir()
    env.cameras["agentview"].set_pose(**av)
    scene = env.scene.visualizer.context._scene
    for node in scene.light_nodes:
        if type(node.light).__name__ != "DirectionalLight":
            continue
        pose = np.array(node.matrix, dtype=np.float32)
        z_up_to_R(-np.asarray(light_dir, dtype=np.float32), out=pose[:3, :3])
        scene.set_pose(node, pose)
