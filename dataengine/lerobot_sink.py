"""EpisodeSink backed by a LeRobot v3 dataset (LIBERO-compatible Panda schema). Requires `lerobot`."""
from pathlib import Path

STATE_NAMES = ["eef_x", "eef_y", "eef_z", "eef_rx", "eef_ry", "eef_rz", "finger_left", "finger_right"]
ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]


def panda_libero_features(image_size: int = 256) -> dict:
    img = {"dtype": "video", "shape": (image_size, image_size, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": STATE_NAMES},
        "observation.images.agentview": dict(img),
        "observation.images.eye_in_hand": dict(img),
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
    }


class LeRobotSink:
    def __init__(self, repo_id: str, root: str | Path, fps: int = 20, image_size: int = 256):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=panda_libero_features(image_size),
                                             root=root, robot_type="panda", use_videos=True)

    def add_frame(self, frame: dict) -> None:
        self.dataset.add_frame(frame)

    def save_episode(self) -> None:
        self.dataset.save_episode()

    def discard_episode(self) -> None:
        if self.dataset.has_pending_frames():
            self.dataset.clear_episode_buffer()

    def finalize(self) -> None:
        self.dataset.finalize()
