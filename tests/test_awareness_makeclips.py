import io
import pytest

Image = pytest.importorskip("PIL.Image")
from awareness.make_clips import make_clips
from awareness.procedure import ProcedureSpec

PROC = ProcedureSpec.from_yaml("configs/procedures/coronary_angiography.yaml")


def png():
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, "PNG")
    return buf.getvalue()


def test_makes_per_step_clips_with_unique_ids_and_cycles_references(tmp_path):
    prompts = []
    clips = make_clips(lambda img, p: prompts.append(p) or png(), [png(), png()], PROC, tmp_path,
                       per_step=2, frames=2, max_images=100, start=1, workers=1, size=16)
    assert len(clips) == 2 * len(PROC.steps) and len(prompts) == len(clips) * 2
    assert len({c.clip_id for c in clips}) == len(clips) and clips[0].clip_id == "access_prep_001"


def test_image_cap_is_checked_before_anything_is_generated(tmp_path):
    calls = []
    with pytest.raises(ValueError, match="max_images"):
        make_clips(lambda img, p: calls.append(1) or png(), [png()], PROC, tmp_path, per_step=3, frames=4, max_images=10)
    assert not calls


def test_steps_without_a_tool_do_not_reach_for_an_object(tmp_path):
    prompts = []
    make_clips(lambda img, p: prompts.append(p) or png(), [png()], PROC, tmp_path, per_step=1, frames=4,
               max_images=100, only=["sheath_insertion"], workers=1, size=16)
    assert not any("reaches a gloved hand toward" in p for p in prompts) and any("inserts the introducer sheath" in p for p in prompts)


def test_no_tool_steps_put_a_patient_and_their_own_cue_in_the_prompt(tmp_path):
    prompts = []
    make_clips(lambda img, p: prompts.append(p) or png(), [png()], PROC, tmp_path, per_step=1, frames=4,
               max_images=100, only=["closure"], workers=1, size=16)
    assert "patient" in prompts[0] and any("gauze pad" in p for p in prompts[1:])
