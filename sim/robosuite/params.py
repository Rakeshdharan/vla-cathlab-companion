"""Expert parameters tuned for robosuite OSC_POSE (measured)."""
from dataengine.expert import ExpertParams

ROBOSUITE_EXPERT_PARAMS = ExpertParams(
    # Releasing 5 cm above the tray floor (not 2 cm) avoids the stretched low pose where OSC stalled ~1 cm short
    # with the 0.3 kg bottle.
    release_height=0.05,
    # robosuite's grip site is ~11 mm above the Panda fingertips; lower and a finger drags on the table and jams.
    min_grasp_tcp_z=0.013,
)
