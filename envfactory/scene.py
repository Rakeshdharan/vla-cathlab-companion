from contracts import SceneSpec

# This module is our "Environment Factory".
# It automatically builds 3D virtual rooms (scenes) for our robot to practice in.

def from_prompt(prompt: str) -> SceneSpec:
    """
    Takes a text description (e.g., "cathlab side table with 8 instruments")
    and automatically builds a 3D scene matching that description.
    """
    print(f"Building a scene from text prompt: '{prompt}'")
    # Using simple logic to parse the prompt (placeholder)
    items = []
    if "instruments" in prompt:
        items.append("instruments")
    if "tray" in prompt:
        items.append("tray")
        
    return SceneSpec(name="generated_scene", items=items)

def from_yaml(path: str) -> SceneSpec:
    """
    Loads a manually designed scene from a configuration file.
    """
    print(f"Loading scene from file: {path}")
    return SceneSpec(name="yaml_scene", items=["from_file"])
