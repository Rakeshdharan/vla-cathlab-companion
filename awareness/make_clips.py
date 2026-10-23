import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from awareness.clipgen import Clip, Edit, generate_clip
from awareness.procedure import ProcedureSpec


def make_clips(edit: Edit, references: list[bytes], procedure: ProcedureSpec, out_dir: str | Path, per_step: int = 2,
               frames: int = 4, max_images: int = 48, start: int = 0, workers: int = 3, size: int = 224,
               only: list[str] | None = None) -> list[Clip]:
    """Generate `per_step` clips for each step, cycling the reference frames. `max_images` is checked first."""
    steps = [s.step_id for s in procedure.steps if not only or s.step_id in only]
    total = len(steps) * per_step * frames
    if total > max_images:
        raise ValueError(f"{total} images requested, over max_images={max_images}")
    jobs = [(step, start + k, references[(i * per_step + k) % len(references)])
            for i, step in enumerate(steps) for k in range(per_step)]

    def one(job):
        step, n, ref = job
        return generate_clip(edit, ref, procedure, step, out_dir, clip_id=f"{step}_{n:03d}", frames=frames, size=size)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


def main() -> None:
    from PIL import Image
    from awareness.clipgen import openai_editor

    ap = argparse.ArgumentParser(description="Generate synthetic demo clips (OpenAI image edits, low quality)")
    ap.add_argument("--refs", nargs="+", required=True, help="reference frames (PNG); kept outside git")
    ap.add_argument("--out", default="var/awareness/clips")
    ap.add_argument("--procedure", default="configs/procedures/coronary_angiography.yaml")
    ap.add_argument("--per-step", type=int, default=2)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=48)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", default="gpt-image-1-mini", help="OpenAI image model (the mini model is the cheapest)")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except ImportError:
        pass
    refs = []
    for r in a.refs:
        buf = io.BytesIO()
        Image.open(r).convert("RGB").resize((1024, 1024), Image.LANCZOS).save(buf, "PNG")
        refs.append(buf.getvalue())
    clips = make_clips(openai_editor(model=a.model), refs, ProcedureSpec.from_yaml(a.procedure), a.out, a.per_step, a.frames,
                       a.max_images, a.start, only=a.only)
    print(f"{len(clips)} clips, {len(clips) * a.frames} images -> {a.out}")


if __name__ == "__main__":
    main()
