from pathlib import Path
import pytest

pytest.importorskip("lerobot")

from contracts import TaskSpec, load_yaml  # noqa: E402
from dataengine.fake_sim import FakeExpertSim  # noqa: E402
from dataengine.generate import generate_scripted  # noqa: E402
from dataengine.lerobot_sink import LeRobotSink  # noqa: E402
from dataengine.variation import VariationSpec, train_configs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SPEC = VariationSpec.load(ROOT / "configs/variation/cathlab_v1.yaml")
TASK = load_yaml(ROOT / "configs/tasks/cathlab_pick.yaml", TaskSpec)


def test_lerobot_sink_writes_loadable_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"
    sink = LeRobotSink("local/cathlab_pick_fake", root, fps=20, image_size=64)
    report, _ = generate_scripted(FakeExpertSim(SPEC, image_size=64), SPEC, TASK, train_configs(SPEC)[:2], sink,
                                  root, "fake", "cathlab/pick_fake", "v1", "test")
    ds = LeRobotDataset("local/cathlab_pick_fake", root=root)
    assert ds.num_episodes == report.succeeded == 2 and ds.fps == 20
    item = ds[0]
    assert tuple(item["action"].shape) == (7,) and tuple(item["observation.state"].shape) == (8,)
    assert tuple(item["observation.images.agentview"].shape) == (3, 64, 64)
    assert item["task"].startswith(("Hand me", "Please pass", "Can you give"))
