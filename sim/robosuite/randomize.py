"""Apply an EpisodeConfig to CathlabPickPlace: poses, object/table colours, mass, friction and lighting."""
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


def apply_episode_config(env, spec: VariationSpec, cfg: EpisodeConfig) -> None:
    """Call right after env.reset(), then settle. Model edits persist with hard_reset=False, so every value is set
    from the env's nominal values, never scaled in place."""
    m = env.sim.model
    palette = spec.obj_colors[cfg.obj_color_id]
    for name, obj in env.objects.items():
        for g in obj.visual_geoms:
            m.geom_rgba[m.geom_name2id(g)] = palette[name]
        for g in obj.contact_geoms:
            m.geom_friction[m.geom_name2id(g), 0] = cfg.friction
        mass, inertia = env.nominal_inertial[name]
        body = env.obj_body_ids[name]
        m.body_mass[body] = mass * cfg.mass_scale
        m.body_inertia[body] = inertia * cfg.mass_scale

    table = m.geom_name2id("table_visual")
    m.geom_matid[table] = -1  # plain colour instead of the ceramic texture
    m.geom_rgba[table] = spec.table_colors[cfg.table_color_id]

    tint = cfg.light_intensity * kelvin_to_rgb(cfg.light_temp_k)
    nominal = env.nominal_lights
    m.light_diffuse[:] = nominal["diffuse"] * tint
    m.light_specular[:] = nominal["specular"] * tint
    headlight = m._model.vis.headlight
    headlight.ambient[:] = nominal["head_ambient"] * tint
    headlight.diffuse[:] = nominal["head_diffuse"] * tint

    env.set_object_poses(object_poses(spec, cfg))
