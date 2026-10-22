"""Apply an EpisodeConfig to CathlabPickPlace: object poses (later: colours, physics, lighting)."""
from dataengine.variation import EpisodeConfig, VariationSpec, object_poses


def apply_episode_config(env, spec: VariationSpec, cfg: EpisodeConfig) -> None:
    """Call right after env.reset(); settle afterwards."""
    env.set_object_poses(object_poses(spec, cfg))
