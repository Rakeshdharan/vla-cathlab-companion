"""GapSpec: what the learning loop asks a data generator to produce more of."""
from pydantic import BaseModel


class GapSpec(BaseModel):
    object: str
    region: str
    clutter: str
    stage: str
