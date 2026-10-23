"""LeRobotPolicyAdapter: key mapping and output shape/clip without lerobot (GPU-free), plus one real checkpoint."""
import numpy as np
import pytest
from train.lerobot_adapter import LeRobotPolicyAdapter, to_actions, to_frame


def obs():
    rng = np.random.default_rng(0)
    return {"state": np.arange(8, dtype=np.float32),
            "agentview": rng.integers(0, 256, (256, 256, 3), dtype=np.uint8),
            "eye_in_hand": np.full((256, 256, 3), 255, dtype=np.uint8)}


def test_to_frame_uses_dataset_keys_and_chw_unit_images():
    o = obs()
    f = to_frame(o, "Hand me the vial")
    assert set(f) == {"task", "observation.state", "observation.images.agentview", "observation.images.eye_in_hand"}
    assert f["task"] == ["Hand me the vial"] and f["observation.state"].shape == (1, 8)
    img = f["observation.images.agentview"]
    assert img.shape == (1, 3, 256, 256) and img.dtype == np.float32
    np.testing.assert_allclose(img[0, :, 10, 20], o["agentview"][10, 20] / 255.0)
    assert f["observation.images.eye_in_hand"].max() == 1.0


def test_to_actions_trims_steps_and_dims_and_clips():
    chunk = np.linspace(-3, 3, 50 * 20, dtype=np.float32).reshape(1, 50, 20)
    a = to_actions(chunk, n_action_steps=30, action_dim=7)
    assert a.shape == (30, 7) and a.dtype == np.float32 and np.all(np.abs(a) <= 1)


def test_predict_with_fake_inference():
    seen = {}
    adapter = LeRobotPolicyAdapter(n_action_steps=10)

    def fake_infer(frame):
        seen.update(frame)
        return np.full((1, 50, 7), 2.0, dtype=np.float32)

    adapter._infer = fake_infer
    a = adapter.predict(obs(), "Hand me the syringe")
    assert a.shape == (10, 7) and np.all(a == 1.0) and seen["task"] == ["Hand me the syringe"]
    adapter.load("")   # engines call load("") on factory-built policies: no-op once loaded


def test_predict_before_load_is_an_error():
    with pytest.raises(RuntimeError, match="load"):
        LeRobotPolicyAdapter().predict(obs(), "")


def test_real_act_checkpoint():
    pytest.importorskip("lerobot")
    adapter = LeRobotPolicyAdapter("Rakeshdharan56/act_panda_pick_v1")
    a = adapter.predict(obs(), "Hand me the vial")
    assert a.shape == (adapter.meta().chunk_size, 7) and a.dtype == np.float32 and np.all(np.abs(a) <= 1)
