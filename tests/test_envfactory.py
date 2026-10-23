from pathlib import Path
from contracts import SceneSpec
from envfactory import from_prompt, from_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_from_prompt_returns_a_valid_scene_spec():
    scene = from_prompt("cathlab side table with 8 instruments and a tray")
    assert isinstance(scene, SceneSpec)
    assert set(scene.objects) == {"instruments", "tray"}
    assert scene.backend == "dummy"


def test_from_prompt_without_known_items_is_empty():
    assert from_prompt("an empty room").objects == {}


def test_from_yaml_loads_the_scene_config():
    scene = from_yaml(ROOT / "configs/scenes/dummy.yaml")
    assert scene.scene_id == "dummy" and "guidewire" in scene.objects
