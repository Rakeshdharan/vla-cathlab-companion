import json
from contracts import EpisodeResult, EvalReport
from dataengine.variation import VariationSpec, eval_configs
from report.failure_report import build_report, cell_of, main, next_run, render_markdown

SPEC_PATH = "configs/variation/cathlab_v1.yaml"
SPEC = VariationSpec.load(SPEC_PATH)


def ep(i, tier, obj, seed, success, stage="grasp", video=None, verdict=None):
    return EpisodeResult(episode=i, seed=seed, object=obj, instruction=f"Hand me the {obj}", success=success,
                         failed_stage=None if success else stage, steps=100, latency_ms_p50=10, latency_ms_p95=20,
                         scene_variant=tier, video_path=video, judge_verdict=verdict)


def rep(engine, episodes):
    return EvalReport(model_version="m@v1", dataset_versions=[], backend=engine, episodes=episodes)


def test_cell_of_rebuilds_the_eval_config():
    cfg = next(c for c in eval_configs(SPEC, "held_out_layout", [5001]) if c.object == "vial")
    cell = cell_of(SPEC, ep(0, "held_out_layout", "vial", 5001, False))
    assert cell == {"tier": "held_out_layout", "object": "vial", "layout_id": cfg.layout_id,
                    "obj_color_id": cfg.obj_color_id, "table_color_id": cfg.table_color_id}
    assert cell_of(SPEC, ep(0, "default", "vial", 1, False)) is None


def test_report_sections_and_next_run():
    rs = rep("robosuite", [ep(0, "in_distribution", "vial", 5000, False, "grasp", video="videos/a.gif"),
                           ep(1, "in_distribution", "vial", 5001, False, "reach"),
                           ep(2, "in_distribution", "bottle", 5000, True),
                           ep(3, "held_out_obj_color", "syringe", 5000, False, "grasp", verdict="wrong object"),
                           ep(4, "in_distribution", "syringe", 5000, False, "lift")])
    ms = rep("maniskill", [ep(0, "in_distribution", "vial", 5000, False, "grasp"),
                           ep(1, "in_distribution", "bottle", 5000, False, "transport")])
    r = build_report({"robosuite": rs, "maniskill": ms}, SPEC, SPEC_PATH, "m@v1", top=10, k=8)
    assert r["overall"]["robosuite"]["n"] == 5 and r["overall"]["robosuite"]["successes"] == 1
    assert r["stage_failures"]["robosuite"]["grasp"] == 2 and r["wrong_object"]["robosuite"] == 1
    gap = {g["tier"]: g for g in r["sim2sim_gap"]}
    assert gap["in_distribution"]["gaps"]["maniskill"] == round(1 / 4 - 0 / 2, 4)
    recs = {(c["object"], c["tiers"][0]): c["recommendation"] for c in r["failing_cells"]}
    assert recs[("vial", "in_distribution")].startswith("generate 8 targeted episodes")
    assert recs[("syringe", "held_out_obj_color")].startswith("widen")
    assert any(c["recommendation"].startswith("check expert params") for c in r["failing_cells"])
    assert r["failing_cells"][0]["object"] == "vial" and r["failing_cells"][0]["video"] == "videos/a.gif"
    nr = next_run(r)
    assert nr["spec"] == SPEC_PATH and nr["k"] == 8 and nr["cells"]
    train = (set(SPEC.layout_split.train), set(SPEC.obj_color_split.train), set(SPEC.table_color_split.train))
    for c in nr["cells"]:
        assert c["layout_id"] in train[0] and c["obj_color_id"] in train[1] and c["table_color_id"] in train[2]
        assert c["k"] == 8
    assert all(c["object"] != "syringe" or c["layout_id"] in train[0] for c in nr["cells"])
    md = render_markdown(r)
    assert md.startswith("# Failure report: m@v1") and "## Sim2sim gap" in md and "## Next run" in md


def test_cli_writes_three_files(tmp_path):
    p = tmp_path / "rs.json"
    p.write_text(rep("robosuite", [ep(0, "in_distribution", "vial", 5000, False)]).model_dump_json())
    out = tmp_path / "out"
    assert main(["--eval", f"robosuite={p}", "--spec", SPEC_PATH, "--model", "m@v1", "--out", str(out)]) == 0
    assert {f.name for f in out.iterdir()} == {"report.json", "report.md", "next_run.json"}
    assert json.loads((out / "next_run.json").read_text())["cells"][0]["object"] == "vial"
