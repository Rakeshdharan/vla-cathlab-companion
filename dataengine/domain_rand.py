from contracts import GapSpec
from dataengine.registry import GenerationReceipt
from dataengine.registry import register_method

# This file contains the "Domain Randomization" data generator.
# Its job is to take existing robot demonstrations and create new practice data
# by randomly tweaking the environment (like changing the lighting, adding random clutter,
# or slightly shaking the camera). This helps the robot learn to not be confused by small changes.

@register_method('domain_rand')
def generate_domain_rand_data(gap: GapSpec | None, n: int) -> GenerationReceipt:
    """
    Generates synthetic training data using domain randomization.
    """
    # 1. Check if we have a specific "gap" to focus on.
    # A gap tells us exactly what the robot is struggling with (e.g., a specific object or lighting).
    if gap:
        print(f"Targeting specific gap: Object={gap.object}, Region={gap.region}, Clutter={gap.clutter}")
        # Here we would configure our randomization to focus on this specific gap.
        # For example, if clutter is "high", we ensure we spawn lots of random objects.
    else:
        print("No specific gap provided. Using general random settings.")

    # 2. Set up the random parameters (poses, clutter, lighting, distractors, camera jitter).
    # (Placeholder: In the real implementation, we will pass these to the engine's randomizer)
    randomization_config = {
        "lighting_variations": True,
        "add_distractor_objects": True,
        "camera_jitter": True,
        "clutter_level": gap.clutter if gap else "medium"
    }
    
    # 3. "Run" the generation process to create 'n' episodes.
    # We would tell the simulator to play back the demonstration 'n' times, 
    # applying our random tweaks each time.
    print(f"Running domain randomization to generate {n} episodes with config: {randomization_config}...")
    
    # 4. Return a "receipt" (GenerationReceipt) confirming what we just created.
    # As per the task requirements, we use a specific version tag for this dataset.
    return GenerationReceipt(
        source='cathlab/handover_dr@v1', 
        num_episodes=n
    )
