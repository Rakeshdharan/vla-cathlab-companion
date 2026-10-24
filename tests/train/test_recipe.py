import json
import shlex

import pytest

from train.recipe import build_argv, load_recipe

# The command that trained Rakeshdharan56/xvla_panda_genesis_v1 (outputs/logs/train_xvla_genesis.sh).
XVLA_GENESIS = """lerobot-train --policy.path=lerobot/xvla-base
  --dataset.repo_id=Rakeshdharan56/pick_genesis --dataset.revision=v1 --dataset.video_backend=pyav
  --rename_map='{"observation.images.agentview": "observation.images.image", "observation.images.eye_in_hand": "observation.images.image2"}'
  --policy.dtype=bfloat16 --policy.action_mode=auto --policy.max_action_dim=20
  --policy.freeze_vision_encoder=false --policy.freeze_language_encoder=false
  --policy.train_policy_transformer=true --policy.train_soft_prompts=true
  --batch_size=32 --steps=7500 --save_freq=2500 --num_workers=16 --seed=1000
  --output_dir=outputs/train/xvla_panda_genesis_v1 --job_name=xvla-pick_genesis_v1-rakesh
  --policy.device=cuda --policy.repo_id=Rakeshdharan56/xvla_panda_genesis_v1
  --wandb.enable=true --wandb.project=cathlab-companion --wandb.mode=offline"""


def flags(argv: list[str]) -> dict[str, str]:
    out = dict(a[2:].split("=", 1) for a in argv[1:])
    if "rename_map" in out:
        out["rename_map"] = json.loads(out["rename_map"])
    return out


def test_xvla_recipe_reproduces_the_genesis_run():
    built = flags(build_argv("xvla", "Rakeshdharan56/pick_genesis@v1", "Rakeshdharan56/xvla_panda_genesis_v1"))
    real = flags(shlex.split(XVLA_GENESIS))
    assert {k: built.get(k) for k in real} == real


def test_smolvla_recipe_matches_v1_and_overrides():
    f = flags(build_argv("smolvla", "Rakeshdharan56/pick_robosuite@v1", "Rakeshdharan56/smolvla_panda_pick_v1"))
    assert f["policy.path"] == "lerobot/smolvla_libero" and f["policy.use_amp"] == "true"
    assert (f["steps"], f["batch_size"]) == ("8500", "32")
    assert f["rename_map"]["observation.images.agentview"] == "observation.images.camera1"
    f = flags(build_argv("xvla", "aravindsuresh16/cathlab_fr3_newton_v1@v1.0.1", "Rakeshdharan56/xvla_fr3_pick_v1",
                         io="fr3_duo_16d", steps=3500))
    assert f["steps"] == "3500" and f["dataset.revision"] == "v1.0.1"
    assert f["rename_map"]["observation.images.right_wrist_camera"] == "observation.images.image3"
    assert f["job_name"] == "xvla-cathlab_fr3_newton_v1_v1.0.1-rakesh"


def test_act_has_no_rename_and_bad_inputs_fail():
    f = flags(build_argv("act", "Rakeshdharan56/pick_robosuite@v1", "Rakeshdharan56/act_panda_pick_v1"))
    assert "rename_map" not in f and f["policy.type"] == "act" and f["steps"] == "50000"
    assert {r.model for r in map(load_recipe, ["act", "smolvla", "xvla"])} == {"act", "smolvla", "xvla"}
    with pytest.raises(ValueError, match="revision"):
        build_argv("xvla", "Rakeshdharan56/pick_genesis", "x/y")
    with pytest.raises(ValueError, match="io"):
        build_argv("smolvla", "x/y@v1", "x/z", io="fr3_duo_16d")


def test_local_dataset_root_matches_the_cross_sim_v2_run():
    f = flags(build_argv("xvla", "Rakeshdharan56/pick_panda_v2", "Rakeshdharan56/xvla_panda_pick_v2",
                         dataset_root="data/pick_panda_v2"))
    assert f["dataset.root"] == "data/pick_panda_v2" and "dataset.revision" not in f
    assert f["job_name"] == "xvla-pick_panda_v2-rakesh" and f["output_dir"] == "outputs/train/xvla_panda_pick_v2"


def test_cli_accepts_the_pipeline_train_stage_argv(capsys):
    from train.recipe import main
    assert main(["--out", "var/runs/r1/train/j1", "--model", "xvla", "--dataset", "x/pick_genesis@v1",
                 "--repo-id", "x/xvla_v9", "--steps", "10", "--io", "panda_libero"]) == 0
    f = flags(shlex.split(capsys.readouterr().out))
    assert f["output_dir"] == "var/runs/r1/train/j1" and f["steps"] == "10" and f["policy.path"] == "lerobot/xvla-base"
