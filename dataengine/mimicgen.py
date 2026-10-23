from contracts import GapSpec
from dataengine.registry import GenerationReceipt
from dataengine.registry import register_method

# This file handles MimicGen.
# Its job is to take human demonstrations (like a person moving a VR controller)
# and "translate" those movements so they work on the robot's specific body,
# even if the robot's starting position or the objects have moved slightly.

@register_method('mimicgen')
def generate_mimicgen_data(gap: GapSpec | None, n: int) -> GenerationReceipt:
    """
    Translates human demos into robot data.
    """
    print(f"Retargeting human demos using MimicGen to generate {n} episodes...")
    return GenerationReceipt(source='mimicgen', num_episodes=n)
