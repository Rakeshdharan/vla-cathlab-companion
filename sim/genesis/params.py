"""Expert parameters for the Genesis delta-EE controller (see docs/decisions.md for the measurements)."""
from dataengine.expert import ExpertParams

GENESIS_EXPERT_PARAMS = ExpertParams(
    release_height=0.065,
    min_grasp_tcp_z=0.013,
)
