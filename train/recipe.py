"""Training recipes as code: train/recipes/<model>.yaml -> the exact `lerobot-train` command (run in the lerobot env).

    python -m train.recipe xvla --dataset Rakeshdharan56/pick_genesis@v1 --repo-id Rakeshdharan56/xvla_panda_genesis_v1
    python -m train.recipe xvla --dataset aravindsuresh16/cathlab_fr3_newton_v1@v1.0.1 --io fr3_duo_16d --steps 3500 \\
        --repo-id Rakeshdharan56/xvla_fr3_pick_v1 --run

Prints the command (shell-quoted) unless --run. Datasets are always pinned to a revision. The recipes reproduce the
published checkpoints (tests/train/test_recipe.py checks the xvla_panda_genesis_v1 command flag for flag).
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml
from pydantic import BaseModel

RECIPES = Path(__file__).resolve().parent / "recipes"


class Recipe(BaseModel):
    model: str
    policy: dict
    rename: dict[str, dict[str, str]] = {}
    train: dict


def load_recipe(model: str) -> Recipe:
    path = RECIPES / f"{model}.yaml"
    if not path.is_file():
        raise ValueError(f"no recipe {path.name}; have {sorted(p.stem for p in RECIPES.glob('*.yaml'))}")
    return Recipe.model_validate(yaml.safe_load(path.read_text()))


def _value(v) -> str:
    return str(v).lower() if isinstance(v, bool) else str(v)


def build_argv(model: str, dataset: str, repo_id: str, io: str = "panda_libero", steps: int | None = None,
               owner: str = "rakesh", output_dir: str | None = None, wandb_mode: str = "offline", seed: int = 1000,
               device: str = "cuda", dataset_root: str | None = None, extra: list[str] | None = None) -> list[str]:
    """dataset = '<repo>@<revision>' (or '<repo>' with a local dataset_root); repo_id = the checkpoint's Hub repo (also
    names the output dir)."""
    recipe = load_recipe(model)
    repo, sep, revision = dataset.partition("@")
    if not dataset_root and (not sep or not revision):
        raise ValueError(f"pin the dataset revision: '{dataset}' -> '<repo>@<revision>'")
    if recipe.rename and io not in recipe.rename:
        raise ValueError(f"recipe {model} has no camera rename for io '{io}' (have {sorted(recipe.rename)})")
    train = {**recipe.train, **({"steps": steps} if steps else {})}
    argv = ["lerobot-train"]
    argv += [f"--policy.{k}={_value(v)}" for k, v in recipe.policy.items()]
    argv += [f"--dataset.repo_id={repo}"] + ([f"--dataset.revision={revision}"] if revision else [])
    argv += ([f"--dataset.root={dataset_root}"] if dataset_root else []) + ["--dataset.video_backend=pyav"]
    if recipe.rename:
        argv.append(f"--rename_map={json.dumps(recipe.rename[io])}")
    argv += [f"--{k}={_value(v)}" for k, v in train.items()]
    argv += [f"--seed={seed}", f"--output_dir={output_dir or 'outputs/train/' + repo_id.split('/')[-1]}",
             f"--job_name={model}-{repo.split('/')[-1]}{'_' + revision if revision else ''}-{owner}", f"--policy.device={device}",
             f"--policy.repo_id={repo_id}", "--wandb.enable=true", "--wandb.project=cathlab-companion",
             f"--wandb.mode={wandb_mode}"]
    return argv + list(extra or [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="recipe: " + ", ".join(sorted(p.stem for p in RECIPES.glob("*.yaml"))))
    ap.add_argument("--model", dest="model_opt", help="same as the positional (used by the PIPELINE train stage)")
    ap.add_argument("--dataset", required=True, help="<hf-repo>@<revision> (revision optional with --dataset-root)")
    ap.add_argument("--dataset-root", help="local LeRobot dataset folder, e.g. data/pick_panda_v2")
    ap.add_argument("--repo-id", required=True, help="checkpoint repo, e.g. Rakeshdharan56/xvla_panda_genesis_v1")
    ap.add_argument("--io", default="panda_libero")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--owner", default="rakesh")
    ap.add_argument("--output-dir", "--out", dest="output_dir")
    ap.add_argument("--wandb-mode", default="offline")
    ap.add_argument("--run", action="store_true", help="run it instead of printing it")
    args, extra = ap.parse_known_args(argv)
    model = args.model_opt or args.model
    if not model:
        ap.error("give the recipe as the positional argument or --model")
    cmd = build_argv(model, args.dataset, args.repo_id, args.io, args.steps, args.owner, args.output_dir,
                     args.wandb_mode, dataset_root=args.dataset_root, extra=extra)
    if not args.run:
        print(shlex.join(cmd))
        return 0
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
