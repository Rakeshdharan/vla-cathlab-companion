from pydantic import BaseModel

# GapSpec represents a scenario where the robot is struggling.
# It tells our data generator exactly what kind of practice data to create.
class GapSpec(BaseModel):
    object: str   # The object the robot is struggling with (e.g., "guidewire")
    region: str   # Where it's happening (e.g., "table")
    clutter: str  # How messy the environment is (e.g., "high")
    stage: str    # What part of the task is failing (e.g., "grasp")

# DatasetManifest is the "receipt" or summary of the data we just generated.
class DatasetManifest(BaseModel):
    source: str       # Which method created this data (e.g., "domain_rand")
    num_episodes: int # How many practice examples were created

# SceneSpec defines the layout of the virtual practice room.
class SceneSpec(BaseModel):
    name: str
    items: list[str]

# PolicyAdapter is the standard shape that any robot "brain" (whether AI or scripted) must follow.
class PolicyAdapter:
    def load(self):
        pass
    def predict(self, obs, instruction):
        return (None, None)
    def meta(self):
        return {}
