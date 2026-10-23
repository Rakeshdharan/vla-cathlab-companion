from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import yaml


@dataclass
class LabeledClip:
    clip_id: str
    step_id: str
    tools: list[str]
    frames: list[np.ndarray]

    def stream(self, fps: float = 1.0) -> Iterator[tuple[float, np.ndarray]]:
        for i, frame in enumerate(self.frames):
            yield i / fps, frame


def load_clip(clip_dir: str | Path) -> LabeledClip:
    from PIL import Image

    d = Path(clip_dir)
    labels = yaml.safe_load((d / "labels.yaml").read_text())
    frames = [np.asarray(Image.open(d / name).convert("RGB"), dtype=np.uint8) for name in labels["frames"]]
    return LabeledClip(labels["clip_id"], labels["step_id"], list(labels.get("tools", [])), frames)


def load_clips(root: str | Path) -> list[LabeledClip]:
    return [load_clip(p.parent) for p in sorted(Path(root).glob("*/labels.yaml"))]
