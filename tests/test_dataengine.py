import pytest
from contracts import GapSpec
from dataengine.registry import GenerationReceipt, generate, list_methods, register_method

# Create a fake generator just for testing purposes
@register_method("dummy_gen")
def dummy_generator(gap: GapSpec | None, n: int) -> GenerationReceipt:
    return GenerationReceipt(source="dummy_gen", num_episodes=n)

# Test 1: Make sure the system crashes properly if we ask for a tool that doesn't exist
def test_generate_unknown_method():
    with pytest.raises(ValueError):
        generate(None, 10, "unknown")

# Test 2: Make sure a valid generation request works and returns the correct receipt
def test_generate_success():
    # Simulate the loop telling us: "The robot is failing to grasp the guidewire on a highly cluttered table"
    gap = GapSpec(object="guidewire", region="table", clutter="high", stage="grasp")
    
    # Generate 10 examples using our fake generator
    manifest = generate(gap, 10, "dummy_gen")
    
    # Check if the receipt matches our request
    assert manifest.source == "dummy_gen"
    assert manifest.num_episodes == 10

# Test 3: Make sure our phonebook correctly lists available tools
def test_list_methods():
    methods = list_methods()
    assert "dummy_gen" in methods
