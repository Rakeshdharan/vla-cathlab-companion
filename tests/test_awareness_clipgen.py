import io
from pathlib import Path
import pytest
import yaml

Image = pytest.importorskip("PIL.Image")
from awareness.clipgen import generate_clip
from awareness.procedure import ProcedureSpec

ROOT = Path(__file__).resolve().parents[1]
PROC = ProcedureSpec.from_yaml(ROOT / "configs/procedures/coronary_angiography.yaml")


def png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


def test_clip_chains_edits_and_writes_labels(tmp_path):
    calls = []

    def edit(image: bytes, prompt: str) -> bytes:
        calls.append((image, prompt))
        return png((len(calls) * 40, 0, 0))

    ref = png((255, 255, 255))
    clip = generate_clip(edit, ref, PROC, "access_prep", tmp_path, clip_id="c0", frames=4, size=32)

    assert len(calls) == 4
    assert calls[0][0] == ref and calls[1][0] != ref
    assert "vial" in calls[1][1]
    assert sorted(p.name for p in (tmp_path / "c0").glob("*.png")) == [f"{i:02d}.png" for i in range(4)]
    assert Image.open(tmp_path / "c0/00.png").size == (32, 32)
    labels = yaml.safe_load((tmp_path / "c0/labels.yaml").read_text())
    assert labels["step_id"] == "access_prep" and labels["source"] == "synthetic"
    assert labels["tools"] == ["vial"] and clip.step_id == "access_prep"
