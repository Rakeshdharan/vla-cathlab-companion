"""Append the cathlab training section to a checkpoint's HF model card (idempotent: replaces its own section).

    python -m train.model_card Rakeshdharan56/smolvla_panda_maniskill_v1 --dataset Rakeshdharan56/pick_maniskill@v1 \
        --base lerobot/smolvla_libero --steps 8500 --batch 32 --wall "1:54" --loss 0.011 --vram 28.4
"""
import argparse

MARKER = "<!-- cathlab-training -->"


def section(a: argparse.Namespace) -> str:
    rows = [("Dataset", f"`{a.dataset}`"), ("Initialised from", f"`{a.base}`"), ("Steps × batch", f"{a.steps:,} × {a.batch}"),
            ("GPU", a.gpu), ("Wall time", a.wall), ("Final train loss", a.loss), ("VRAM peak", f"{a.vram} GB")]
    if a.note:
        rows.append(("Note", a.note))
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (f"{MARKER}\n## Training (cathlab-companion)\n\n| | |\n|---|---|\n{table}\n\n"
            "Code, eval and recipes: https://github.com/Rakeshdharan/vla-cathlab-companion. "
            "Evaluate with `python -m engines.<robosuite|maniskill>.cli eval --policy lerobot:<this repo>`.\n")


def with_section(card: str, new: str) -> str:
    return (card.split(MARKER)[0].rstrip() + "\n\n" + new) if MARKER in card else card.rstrip() + "\n\n" + new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    for flag in ("--dataset", "--base", "--wall", "--loss", "--vram"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--gpu", default="NVIDIA RTX 6000 Ada (48 GB)")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, hf_hub_download
    card = open(hf_hub_download(a.repo, "README.md")).read()
    HfApi().upload_file(path_or_fileobj=with_section(card, section(a)).encode(), path_in_repo="README.md",
                        repo_id=a.repo, commit_message="Model card: cathlab training details")
    print(f"updated https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
