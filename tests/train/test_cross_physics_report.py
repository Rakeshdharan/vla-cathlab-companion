from contracts import EpisodeResult, EvalReport
from train.cross_physics_report import load_reports, main, render, summarize


def report(model: str, backend: str, successes: list[bool]) -> EvalReport:
    eps = [EpisodeResult(episode=i, seed=5000 + i, object="vial", instruction="Hand me the vial", success=s,
                         failed_stage=None if s else "grasp", steps=100, latency_ms_p50=10.0, latency_ms_p95=20.0,
                         scene_variant="in_distribution" if i % 2 else "all_held_out")
           for i, s in enumerate(successes)]
    return EvalReport(model_version=model, dataset_versions=[], backend=backend, episodes=eps)


def write(root, eng, model, successes):
    d = root / eng / model
    d.mkdir(parents=True)
    (d / "eval_report.json").write_text(report(model, eng, successes).model_dump_json())


def test_table_marks_in_domain_and_counts_failures(tmp_path):
    write(tmp_path, "robosuite", "smolvla_panda_pick_v1", [True, True, False, True])
    write(tmp_path, "maniskill", "smolvla_panda_pick_v1", [False, False, True, False])
    reports = load_reports(tmp_path)
    md = render(reports)
    assert "75% (" in md and "25% (" in md and md.count("**in-domain**") == 1
    assert "| robosuite | SmolVLA (robosuite) | 0 | 1 | 0 | 0 | 0 | 10 / 20 |" in md
    s = summarize(reports)["maniskill/smolvla_panda_pick_v1"]
    assert s["trained_on"] == "robosuite" and s["failed_stage"]["grasp"] == 3 and s["overall"]["n"] == 4


def test_main_writes_md_and_json(tmp_path):
    write(tmp_path / "evals", "maniskill", "expert", [True] * 3)
    main(["--evals", str(tmp_path / "evals"), "--out", str(tmp_path / "out/cross")])
    assert (tmp_path / "out/cross.md").exists() and (tmp_path / "out/cross.json").exists()


def test_model_card_section_is_idempotent():
    import argparse
    from train.model_card import MARKER, section, with_section
    a = argparse.Namespace(dataset="d@v1", base="b", steps=8500, batch=32, gpu="g", wall="1:54", loss="0.011",
                           vram="28.4", note="")
    once = with_section("# card\n", section(a))
    twice = with_section(once, section(a))
    assert once == twice and once.count(MARKER) == 1 and "8,500 × 32" in once
