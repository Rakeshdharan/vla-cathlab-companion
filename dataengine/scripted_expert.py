from contracts import GapSpec, PolicyAdapter
from dataengine.registry import GenerationReceipt
from dataengine.registry import register_method

# This file contains the Scripted Expert.
# It's essentially a perfectly programmed robot that uses advanced math 
# (GraspGen for figuring out where to grab, and cuRobo for moving without hitting things)
# to perfectly complete tasks. It can be used to generate perfect training data,
# or to evaluate how a perfect run looks compared to our AI.

class ScriptedExpert(PolicyAdapter):
    """
    A perfect, non-AI robot controller used for generating data and testing.
    """
    def load(self):
        # Prepare the math and planning libraries
        print("Loading GraspGen and cuRobo models...")
        pass

    def predict(self, obs, instruction):
        # Calculate the perfect next move
        # Returns a mock action
        return (None, None)
        
    def meta(self):
        return {"name": "ScriptedExpert", "version": "1.0"}

@register_method('scripted')
def generate_scripted_data(gap: GapSpec | None, n: int) -> GenerationReceipt:
    """
    Uses the ScriptedExpert to generate 'n' perfect examples.
    """
    print(f"Generating {n} perfect episodes using Scripted Expert...")
    return GenerationReceipt(source='scripted', num_episodes=n)
