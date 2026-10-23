from contracts import GapSpec
from dataengine.registry import GenerationReceipt
from dataengine.registry import register_method

# Generator 4: Teleoperation (Optional)
# Data created by a human manually controlling the robot.
@register_method('teleop')
def teleop_generator(gap: GapSpec | None, n: int) -> GenerationReceipt:
    # TODO: Implement teleop logic here later
    return GenerationReceipt(source='teleop', num_episodes=n)

# Generator 5: Cosmos Transfer (Optional)
# Advanced experimental data generation method.
@register_method('cosmos')
def cosmos_generator(gap: GapSpec | None, n: int) -> GenerationReceipt:
    # TODO: Implement cosmos logic here later
    return GenerationReceipt(source='cosmos', num_episodes=n)
