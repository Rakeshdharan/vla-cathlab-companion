from contracts import GapSpec, DatasetManifest
from dataengine import register_method

# Generator 4: Teleoperation (Optional)
# Data created by a human manually controlling the robot.
@register_method('teleop')
def teleop_generator(gap: GapSpec | None, n: int) -> DatasetManifest:
    # TODO: Implement teleop logic here later
    return DatasetManifest(source='teleop', num_episodes=n)

# Generator 5: Cosmos Transfer (Optional)
# Advanced experimental data generation method.
@register_method('cosmos')
def cosmos_generator(gap: GapSpec | None, n: int) -> DatasetManifest:
    # TODO: Implement cosmos logic here later
    return DatasetManifest(source='cosmos', num_episodes=n)
