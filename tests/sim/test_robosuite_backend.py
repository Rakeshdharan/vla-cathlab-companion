import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("robosuite")

from contracts import STAGES, SceneSpec, StepInfo  # noqa: E402
from sim.robosuite.backend import RobosuiteBackend  # noqa: E402

# SceneSpec.backend has no "robosuite" literal yet and the backend ignores the scene, so skip validation.
SCENE = SceneSpec.model_construct(scene_id="cathlab_v1", backend="robosuite", asset="", objects={},
                                  cameras=["agentview", "eye_in_hand"])


@pytest.fixture(scope="module")
def backend():
    b = RobosuiteBackend(tier="all_held_out")
    yield b
    b.close()


def test_reset_step_render_follow_the_contract(backend):
    obs = backend.reset(SCENE, seed=1000, target_object="vial")
    assert obs["state"].shape == (8,)
    obs, info = backend.step(np.zeros(7, dtype=np.float32))
    assert isinstance(info, StepInfo) and set(info.stage_flags) == set(STAGES) and not info.success
    assert backend.render().shape == (256, 256, 3)


def test_seed_selects_the_shared_eval_config(backend):
    cfg = backend.config_for(1001, "syringe")
    assert cfg.tier == "all_held_out" and cfg.layout_id in (8, 9) and cfg.object == "syringe"
    backend.reset(SCENE, seed=1001, target_object="syringe")
    assert backend.sim.cfg == cfg


def test_rejects_unknown_target_and_tier(backend):
    with pytest.raises(ValueError):
        backend.config_for(1, "scalpel")
    with pytest.raises(ValueError):
        RobosuiteBackend(tier="nope", expert_sim=backend.sim)
