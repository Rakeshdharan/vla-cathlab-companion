"""Apply an EpisodeConfig to CathlabPickEnv (CPU sim, one env): object/table colours, mass, friction, lighting, poses."""
import numpy as np
import sapien
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses
from sim.maniskill.env import NOMINAL_AMBIENT, NOMINAL_DIRECTIONAL
from sim.robosuite.randomize import kelvin_to_rgb


def _entity(actor):
    return actor._objs[0]


def set_color(actor, rgba) -> None:
    for shape in _entity(actor).find_component_by_type(sapien.render.RenderBodyComponent).render_shapes:
        shape.material.set_base_color(list(rgba))


def rigid_body(actor) -> sapien.physx.PhysxRigidDynamicComponent:
    return _entity(actor).find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)


def nominal_inertial(env) -> dict[str, tuple[float, np.ndarray]]:
    """Mass and inertia as built, cached on the env so randomisation never compounds."""
    if not hasattr(env, "_nominal_inertial"):
        env._nominal_inertial = {n: (float(rigid_body(a).mass), np.array(rigid_body(a).inertia))
                                 for n, a in env.objects.items()}
    return env._nominal_inertial


def directional_lights(env) -> list:
    return [c for e in env.scene.sub_scenes[0].entities for c in e.components
            if isinstance(c, sapien.render.RenderDirectionalLightComponent)]


def apply_episode_config(env, spec: VariationSpec, cfg: EpisodeConfig) -> None:
    """Call right after env.reset(), then settle. Every value is set from nominal values, never scaled in place."""
    palette = spec.obj_colors[cfg.obj_color_id]
    for name, (mass, inertia) in nominal_inertial(env).items():
        actor = env.objects[name]
        set_color(actor, palette[name])
        body = rigid_body(actor)
        body.set_mass(mass * cfg.mass_scale)
        body.set_inertia(inertia * cfg.mass_scale)
        material = sapien.physx.PhysxMaterial(static_friction=cfg.friction, dynamic_friction=cfg.friction,
                                              restitution=0.0)
        for shape in body.collision_shapes:
            shape.set_physical_material(material)
    set_color(env.table, spec.table_colors[cfg.table_color_id])

    tint = cfg.light_intensity * kelvin_to_rgb(cfg.light_temp_k)
    env.scene.set_ambient_light((NOMINAL_AMBIENT * tint).tolist())
    for light, (_, color) in zip(directional_lights(env), NOMINAL_DIRECTIONAL):
        light.set_color((np.asarray(color) * tint).tolist())

    env.set_object_poses(object_poses(spec, cfg))
