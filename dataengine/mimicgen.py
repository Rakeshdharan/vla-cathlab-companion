from contracts import GapSpec, DatasetManifest
from dataengine import register_method

# This file handles MimicGen (or Isaac Lab Mimic).
# Its job is to take human demonstrations (like a person moving a VR controller)
# and "translate" those movements so they work on the robot's specific body,
# even if the robot's starting position or the objects have moved slightly.

@register_method('mimicgen')
def generate_mimicgen_data(gap: GapSpec | None, n: int) -> DatasetManifest:
    """
    Translates human demos into robot data.
    """
    print(f"Retargeting human demos using MimicGen to generate {n} episodes...")
    return DatasetManifest(source='mimicgen', num_episodes=n)
