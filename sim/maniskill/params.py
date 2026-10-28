"""Expert parameters tuned for ManiSkill's pd_ee_delta_pose Panda (measured)."""
from dataengine.expert import ExpertParams

MANISKILL_EXPERT_PARAMS = ExpertParams(
    # The side-grasped bottle's release pose over the tray is out of reach below TCP z = 0.21 (IK fails; the arm is
    # fully stretched). 0.10 (TCP 0.21) is the lowest that works; robosuite's compliant OSC got away with 0.05.
    release_height=0.10,
    # ManiSkill's fingertips reach ~16 mm below our TCP (robosuite ~11 mm); at 0.013 they drag on the table and the
    # syringe is missed.
    min_grasp_tcp_z=0.02,
    # A vial dropped from 10 cm can tip and roll; a longer hold + retreat lets it come to rest before the check.
    open_steps=20,
    lift_height=0.30,
)
