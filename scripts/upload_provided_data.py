"""Upload the hackathon-provided datasets and Isaac environment to PRIVATE Hugging Face repos.

Usage (after `hf auth login`):
    python scripts/upload_provided_data.py --namespace <hf-user-or-org> [--src .cache] [--skip-env]

Expects the provided zips unpacked as:
    <src>/extracted/fr3_duo_handover_10ep/        (FR3Duo_Handover_Dataset_lerobot3.zip)
    <src>/extracted/fr3_duo_handover_v9_night_84/ (fr3_duo_handover_84_ep.zip)
    <src>/extracted/fr3_duo_rand500/              (fr3_duo_rand500.zip)
    <src>/C2_Teach-the-robot_VLA/Environment/Environment_Hackathon_2026.zip
Re-running is safe: repos/tags are created only if missing and uploads are resumable.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

CODEBASE_VERSION = "v3.0"  # LeRobot resolves datasets by this git tag
DATASETS = {
    "fr3_duo_handover_10ep": "fr3_duo_handover_10ep",
    "fr3_duo_handover_84ep": "fr3_duo_handover_v9_night_84",
    "fr3_duo_rand490": "fr3_duo_rand500",
}
ENV_REPO = "cathlab_env_isaac"
ENV_ZIP = "C2_Teach-the-robot_VLA/Environment/Environment_Hackathon_2026.zip"

NOTICE = (
    "**Restricted — Siemens Healthineers Hackathon 2026 (Challenge C2) provided material.** "
    "Private to team CathLab Companion. Do not make public or redistribute."
)


def dataset_card(repo_id: str, folder: Path) -> str:
    info = json.loads((folder / "meta/info.json").read_text())
    cams = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    return f"""---
license: other
tags: [LeRobot, fr3_duo, cathlab-companion]
---
# {repo_id}

{NOTICE}

| Field | Value |
|---|---|
| LeRobot codebase | {info["codebase_version"]} |
| Robot | {info["robot_type"]} |
| Episodes / frames | {info["total_episodes"]} / {info["total_frames"]} |
| FPS | {info["fps"]} |
| Action / state dim | {info["features"]["action"]["shape"]} / {info["features"]["observation.state"]["shape"]} |
| Cameras | {", ".join(cams)} |

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("{repo_id}")
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--src", default=".cache", type=Path)
    ap.add_argument("--skip-env", action="store_true")
    args = ap.parse_args()
    api = HfApi()

    for name, folder_name in DATASETS.items():
        repo_id, folder = f"{args.namespace}/{name}", args.src / "extracted" / folder_name
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="dataset")
        api.upload_file(path_or_fileobj=dataset_card(repo_id, folder).encode(), path_in_repo="README.md",
                        repo_id=repo_id, repo_type="dataset", commit_message="Add dataset card")
        api.create_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset", exist_ok=True)
        print(f"done: https://huggingface.co/datasets/{repo_id}")

    if not args.skip_env:
        repo_id = f"{args.namespace}/{ENV_REPO}"
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj=str(args.src / ENV_ZIP), path_in_repo=Path(ENV_ZIP).name,
                        repo_id=repo_id, repo_type="dataset")
        api.upload_file(path_or_fileobj=f"# {repo_id}\n\n{NOTICE}\n\nIsaac Sim / USD environment for the FR3 Duo "
                        f"handover scene. Download: `hf download {repo_id} --repo-type dataset`.\n".encode(),
                        path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
        print(f"done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
