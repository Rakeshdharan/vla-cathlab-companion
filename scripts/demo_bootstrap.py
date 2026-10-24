"""Rebuild the demo state (model registry + failure report) on any machine from the eval reports committed in git.

    uv run python -m scripts.demo_bootstrap            # -> var/registry, var/reports/rep-cathlab-pick-<prod>
    uv run python -m scripts.demo_bootstrap --var /tmp/var --dry-run

var/ is git-ignored, so a fresh clone has no registry; this replays H37 exactly: every Panda checkpoint with eval
reports in reports/robosuite/eval_reports/<engine>/<model>/ is registered as cathlab-pick@vN (training order below),
then PROMOTE_ORDER is promoted by the rule in configs/serve.yaml, and the failure report + next_run.json of the
production version are written for the REPORT tab. Idempotent: registered checkpoints are skipped, a version is only
promoted if it is not production already. New checkpoints: add them to CHECKPOINTS (and PROMOTE_ORDER if they should
be tried for production).
"""
import argparse
from pathlib import Path

from contracts import EvalReport, ModelCard
from dataengine.variation import VariationSpec
from report.failure_report import build_report, write_outputs
from serve.registry import MODEL_NAME, PromotionRule, Registry

ROOT = Path(__file__).resolve().parents[1]
EVALS = "reports/robosuite/eval_reports"
SPEC = "configs/variation/cathlab_v1.yaml"
# (checkpoint, training dataset) in training order = version order.
CHECKPOINTS = [
    ("act_panda_pick_v1", "Rakeshdharan56/pick_robosuite@v1"),
    ("smolvla_panda_pick_v1", "Rakeshdharan56/pick_robosuite@v1"),
    ("xvla_panda_pick_v1", "Rakeshdharan56/pick_robosuite@v1"),
    ("smolvla_panda_maniskill_v1", "Rakeshdharan56/pick_maniskill@v1"),
    ("xvla_panda_maniskill_v1", "Rakeshdharan56/pick_maniskill@v1"),
    ("xvla_panda_genesis_v1", "Rakeshdharan56/pick_genesis@v1"),
    ("xvla_panda_pick_v2", "Rakeshdharan56/pick_panda_v2 (local merge of robosuite + ManiSkill + Genesis v1)"),
]
# Tried in this order; the rule decides (ACT first as the baseline, then the rest by training order).
PROMOTE_ORDER = [c for c, _ in CHECKPOINTS]


def eval_paths(ckpt: str) -> dict[str, str]:
    return {p.parent.parent.name: p.relative_to(ROOT).as_posix()
            for p in sorted((ROOT / EVALS).glob(f"*/{ckpt}/eval_report.json"))}


def bootstrap(var: Path, dry_run: bool = False) -> dict:
    reg = Registry(var / "registry", repo_root=ROOT)
    rule = PromotionRule.from_yaml(ROOT / "configs/serve.yaml")
    known = {v.card.ckpt_uri: v.card.version for v in reg.list_versions(MODEL_NAME)}
    log = []
    for ckpt, dataset in CHECKPOINTS:
        uri = f"Rakeshdharan56/{ckpt}"
        evals = eval_paths(ckpt)
        if uri in known or "robosuite" not in evals:
            log.append(f"skip {ckpt}: " + ("registered as " + known[uri] if uri in known else "no robosuite eval yet"))
            continue
        version = f"v{len(known) + 1}"
        if not dry_run:
            reg.register(ModelCard(model_name=MODEL_NAME, version=version, adapter="lerobot", ckpt_uri=uri,
                                   dataset_versions=[dataset], eval_report_path=evals["robosuite"]), evals)
        known[uri] = version
        log.append(f"register {MODEL_NAME}@{version} = {ckpt} with {sorted(evals)}")
    for ckpt in PROMOTE_ORDER:
        version = known.get(f"Rakeshdharan56/{ckpt}")
        prod = None if dry_run else reg.production(MODEL_NAME)
        if version is None or dry_run or (prod and prod.card.version == version):
            continue
        d = reg.promote(MODEL_NAME, version, rule)
        log.append(f"promote {version} ({ckpt}): {'yes' if d.promoted else 'no'}, {d.reason}")
    prod = None if dry_run else reg.production(MODEL_NAME)
    if prod is not None:
        evals = {e: EvalReport.load(reg.abs_path(p)) for e, p in prod.eval_reports.items()}
        report = build_report(evals, VariationSpec.load(ROOT / SPEC), SPEC, f"{MODEL_NAME}@{prod.card.version}")
        out = var / "reports" / f"rep-{MODEL_NAME}-{prod.card.version}"
        write_outputs(report, out)
        log.append(f"failure report + next_run.json -> {out}")
    return {"production": prod.card.version if prod else None, "log": log}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--var", default=str(ROOT / "var"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    result = bootstrap(Path(args.var), args.dry_run)
    print("\n".join(result["log"]) + f"\nproduction: {MODEL_NAME}@{result['production']}")


if __name__ == "__main__":
    main()
