"""Registry of named data-generation methods (domain_rand, mimicgen, scripted, teleop, cosmos).

The built-in methods are placeholders that return a GenerationReceipt; the working robosuite generator is
scripts/robosuite_generate.py (see docs/tasks/robosuite-collection.md).
"""
from pydantic import BaseModel
from contracts import GapSpec


class GenerationReceipt(BaseModel):
    source: str
    num_episodes: int


_generators: dict = {}


def register_method(name: str):
    def decorator(func):
        _generators[name] = func
        return func
    return decorator


def generate(gap: GapSpec | None, n: int, method: str) -> GenerationReceipt:
    if method not in _generators:
        raise ValueError(f"Unknown generator method: {method}")
    return _generators[method](gap, n)


def list_methods() -> list[str]:
    return list(_generators.keys())


from dataengine import domain_rand, generators, mimicgen, scripted_expert  # noqa: E402,F401  (register built-ins)
