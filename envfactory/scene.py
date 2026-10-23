from pathlib import Path
from contracts import SceneSpec, load_yaml

# This module is our "Environment Factory".
# It automatically builds 3D virtual rooms (scenes) for our robot to practice in.

def from_prompt(prompt: str) -> SceneSpec:
    """
    Takes a text description (e.g., "cathlab side table with 8 instruments")
    and automatically builds a 3D scene matching that description.
    """
    print(f"Building a scene from text prompt: '{prompt}'")
    # Using simple logic to parse the prompt (placeholder); objects map name -> asset ("none" until generated)
    objects = {}
    if "instruments" in prompt:
        objects["instruments"] = "none"
    if "tray" in prompt:
        objects["tray"] = "none"

    return SceneSpec(scene_id="generated_scene", backend="dummy", asset="none", objects=objects,
                     cameras=["agentview", "eye_in_hand"])

def from_yaml(path: str | Path) -> SceneSpec:
    """
    Loads a manually designed scene from a configuration file.
    """
    print(f"Loading scene from file: {path}")
    return load_yaml(path, SceneSpec)
