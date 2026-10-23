import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import yaml
from awareness.procedure import ProcedureSpec

Edit = Callable[[bytes, str], bytes]

STYLE = (
    "Edit this image. Keep the camera angle, room, furniture, trolleys, equipment, colours and flat "
    "simulator-render style exactly the same. No other people. "
)
STAGES = [
    "Add one clinician in blue scrubs, cap, mask and gloves standing at the steel trolley, hands down, looking at the tray.",
    "Same clinician and room. The clinician reaches a gloved hand toward the {tool} on the tray.",
    "Same clinician and room. The clinician holds the {tool} in a gloved hand.",
    "Same clinician and room. The clinician is mid-task: {description}.",
]


STAGES_NO_TOOL = [
    "Add one clinician in blue scrubs, cap, mask and gloves standing beside the examination bed, with a patient lying on the bed covered by a blue drape.",
    "Same clinician, patient and room. The clinician leans over the patient's wrist and begins: {cue}.",
    "Same clinician, patient and room. The clinician is concentrating, both gloved hands busy: {cue}.",
    "Same clinician, patient and room. The clinician is mid-task: {description}; {cue}.",
]


@dataclass
class Clip:
    clip_id: str
    step_id: str
    frames: list[Path]


def openai_editor(model: str = "gpt-image-1", quality: str = "low") -> Edit:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def edit(image: bytes, prompt: str) -> bytes:
        import base64

        buf = io.BytesIO(image)
        buf.name = "frame.png"
        r = client.images.edit(model=model, image=buf, prompt=prompt, size="1024x1024", quality=quality)
        return base64.b64decode(r.data[0].b64_json)

    return edit


def generate_clip(
    edit: Edit,
    reference: bytes,
    procedure: ProcedureSpec,
    step_id: str,
    out_dir: str | Path,
    clip_id: str,
    frames: int = 4,
    size: int = 224,
) -> Clip:
    from PIL import Image

    step = procedure.steps[procedure.index(step_id)]
    tool = step.tools[0] if step.tools else ""
    base = STAGES if tool else STAGES_NO_TOOL
    clip_dir = Path(out_dir) / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    stages = base[:frames] if frames <= len(base) else base + [base[-1]] * (frames - len(base))
    paths, image = [], reference
    for i, stage in enumerate(stages):
        prompt = STYLE + stage.format(tool=tool, cue=step.cue or "working on the task", description=step.description.lower().rstrip("."))
        image = edit(image, prompt)
        path = clip_dir / f"{i:02d}.png"
        Image.open(io.BytesIO(image)).convert("RGB").resize((size, size), Image.LANCZOS).save(path)
        paths.append(path)
    (clip_dir / "labels.yaml").write_text(
        yaml.safe_dump(
            {"clip_id": clip_id, "step_id": step_id, "tools": step.tools, "frames": [p.name for p in paths], "source": "synthetic"},
            sort_keys=False,
        )
    )
    return Clip(clip_id, step_id, paths)
