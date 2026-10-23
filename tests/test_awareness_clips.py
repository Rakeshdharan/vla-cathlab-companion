from pathlib import Path
import numpy as np
import pytest
import yaml
from awareness.clips import load_clip, load_clips

Image = pytest.importorskip("PIL.Image")


def write_clip(root: Path, clip_id: str, step_id: str, n: int = 3) -> Path:
    d = root / clip_id
    d.mkdir(parents=True)
    for i in range(n):
        Image.new("RGB", (16, 16), (i * 50, 0, 0)).save(d / f"{i:02d}.png")
    (d / "labels.yaml").write_text(yaml.safe_dump(
        {"clip_id": clip_id, "step_id": step_id, "tools": ["vial"], "frames": [f"{i:02d}.png" for i in range(n)], "source": "synthetic"}
    ))
    return d


def test_load_clip_returns_ordered_uint8_frames_and_labels(tmp_path):
    clip = load_clip(write_clip(tmp_path, "c0", "access_prep"))
    assert (clip.clip_id, clip.step_id, clip.tools) == ("c0", "access_prep", ["vial"])
    assert len(clip.frames) == 3 and clip.frames[0].shape == (16, 16, 3) and clip.frames[0].dtype == np.uint8
    assert clip.frames[2][0, 0, 0] == 100


def test_clip_stream_yields_time_frame_pairs(tmp_path):
    clip = load_clip(write_clip(tmp_path, "c0", "access_prep"))
    assert [t for t, _ in clip.stream(fps=2.0)] == [0.0, 0.5, 1.0]


def test_load_clips_finds_all_and_sorts(tmp_path):
    write_clip(tmp_path, "b", "closure")
    write_clip(tmp_path, "a", "access_prep")
    assert [c.clip_id for c in load_clips(tmp_path)] == ["a", "b"]
