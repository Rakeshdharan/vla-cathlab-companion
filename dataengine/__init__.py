from contracts import GapSpec, DatasetManifest

# This dictionary acts as a phonebook for all our different data generation methods.
_generators = {}

# This is a special label (decorator) we can put above a function
# to automatically add it to our _generators phonebook.
def register_method(name: str):
    def decorator(func):
        _generators[name] = func
        return func
    return decorator

# The main control panel: tell it what data is missing (gap), how many to make (n),
# and which tool to use (method), and it will generate the data.
def generate(gap: GapSpec | None, n: int, method: str) -> DatasetManifest:
    # Check if the requested tool exists in our phonebook
    if method not in _generators:
        raise ValueError(f"Unknown generator method: {method}")
    
    # Run the requested tool and return its summary/receipt
    return _generators[method](gap, n)

# Returns a list of all the tools we currently have available
def list_methods() -> list[str]:
    return list(_generators.keys())

# Import the generators so they automatically register themselves when this folder is loaded
from . import domain_rand
from . import mimicgen
from . import scripted_expert
from . import generators
