"""Model registry: versions of `cathlab-pick` with their eval reports, the promotion rule, rollback.

    python -m serve.registry list cathlab-pick
    python -m serve.registry register --card card.yaml --eval robosuite=reports/.../eval_report.json
    python -m serve.registry promote cathlab-pick v2 [--force]
    python -m serve.registry rollback cathlab-pick
    python -m serve.registry resolve cathlab-pick@production      # -> {"policy": "lerobot:<ckpt>"}

Storage (git-ignored): <root>/<name>/<version>.yaml and <root>/<name>/production.yaml
({"current": version | null, "history": [{version, at, action, reason}]}).
"""
import argparse
import json
import sys
import time
from pathlib import Path
import yaml
from pydantic import BaseModel
from contracts import EvalReport, ModelCard
from eval.metrics import latency_summary

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "cathlab-pick"


class RegisteredModel(BaseModel):
    card: ModelCard
    eval_reports: dict[str, str]          # engine -> EvalReport JSON path (repo-relative or absolute)
    registered: float


class PromotionRule(BaseModel):
    primary_engine: str = "robosuite"
    latency_budget_ms_p95: float = 500.0
    min_episodes: int = 30

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PromotionRule":
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(**{k: data[k] for k in cls.model_fields if k in data})


class PromotionDecision(BaseModel):
    promoted: bool
    reason: str
    candidate_rate: float
    candidate_ci: tuple[float, float]
    baseline_version: str | None = None
    baseline_rate: float | None = None
    p95_ms: float = 0.0


def policy_string(card: ModelCard) -> str:
    """The engine CLI --policy for a registered card."""
    return "expert" if card.adapter == "expert" else f"lerobot:{card.ckpt_uri}"


class Registry:
    def __init__(self, root: str | Path = "var/registry", repo_root: str | Path | None = None):
        self.root = Path(root)
        self.repo_root = Path(repo_root) if repo_root else ROOT

    # storage -----------------------------------------------------------------------------------
    def _dir(self, name: str) -> Path:
        return self.root / name

    def _path(self, name: str, version: str) -> Path:
        return self._dir(name) / f"{version}.yaml"

    def _save(self, m: RegisteredModel) -> None:
        p = self._path(m.card.model_name, m.card.version)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(m.model_dump(mode="json"), sort_keys=False))

    def _prod(self, name: str) -> dict:
        p = self._dir(name) / "production.yaml"
        return yaml.safe_load(p.read_text()) if p.exists() else {"current": None, "history": []}

    def _save_prod(self, name: str, prod: dict) -> None:
        p = self._dir(name) / "production.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(prod, sort_keys=False))

    def abs_path(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.repo_root / p

    def report(self, m: RegisteredModel, engine: str) -> EvalReport:
        return EvalReport.load(self.abs_path(m.eval_reports[engine]))

    # versions ----------------------------------------------------------------------------------
    def register(self, card: ModelCard, eval_reports: dict[str, str]) -> RegisteredModel:
        if self._path(card.model_name, card.version).exists():
            raise FileExistsError(f"{card.model_name}@{card.version} is already registered")
        for engine, path in eval_reports.items():
            if not self.abs_path(path).is_file():
                raise FileNotFoundError(f"eval report for {engine} not found: {path}")
        if not card.eval_report_path and eval_reports:
            card = card.model_copy(update={"eval_report_path": next(iter(eval_reports.values()))})
        m = RegisteredModel(card=card.model_copy(update={"promoted": False}), eval_reports=eval_reports,
                            registered=time.time())
        self._save(m)
        return m

    def list_versions(self, name: str) -> list[RegisteredModel]:
        models = [RegisteredModel.model_validate(yaml.safe_load(p.read_text()))
                  for p in self._dir(name).glob("*.yaml") if p.name != "production.yaml"]
        return sorted(models, key=lambda m: m.registered)

    def get(self, name: str, version: str) -> RegisteredModel:
        p = self._path(name, version)
        if not p.exists():
            raise KeyError(f"{name}@{version} is not registered")
        return RegisteredModel.model_validate(yaml.safe_load(p.read_text()))

    def production(self, name: str) -> RegisteredModel | None:
        current = self._prod(name)["current"]
        return self.get(name, current) if current else None

    def history(self, name: str) -> list[dict]:
        return self._prod(name)["history"]

    def resolve(self, ref: str) -> RegisteredModel:
        name, sep, version = ref.partition("@")
        if not sep or not name or not version:
            raise ValueError(f"expected <name>@<version|production>, got '{ref}'")
        if version == "production":
            m = self.production(name)
            if m is None:
                raise KeyError(f"{name} has no production version yet")
            return m
        return self.get(name, version)

    def policy_for(self, ref: str) -> str:
        return policy_string(self.resolve(ref).card)

    # promotion ---------------------------------------------------------------------------------
    def _set_current(self, name: str, version: str | None, action: str, reason: str) -> None:
        prod = self._prod(name)
        for m in self.list_versions(name):
            should = m.card.version == version
            if m.card.promoted != should:
                self._save(m.model_copy(update={"card": m.card.model_copy(update={"promoted": should})}))
        prod["current"] = version
        prod["history"].append({"version": version, "at": time.time(), "action": action, "reason": reason})
        self._save_prod(name, prod)

    def promote(self, name: str, version: str, rule: PromotionRule | None = None,
                force: bool = False) -> PromotionDecision:
        rule = rule or PromotionRule()
        cand = self.get(name, version)
        if rule.primary_engine not in cand.eval_reports:
            return PromotionDecision(promoted=False, reason=f"no eval on {rule.primary_engine}",
                                     candidate_rate=0.0, candidate_ci=(0.0, 0.0))
        rep = self.report(cand, rule.primary_engine)
        rate, ci = rep.success_rate(), rep.wilson_ci()
        p95 = latency_summary(rep)["p95_ms"]
        base = self.production(name)
        decision = dict(candidate_rate=rate, candidate_ci=ci, p95_ms=p95,
                        baseline_version=base.card.version if base else None)
        if len(rep.episodes) < rule.min_episodes:
            return PromotionDecision(promoted=False, reason=f"insufficient eval on {rule.primary_engine}: "
                                     f"{len(rep.episodes)} < {rule.min_episodes} episodes", **decision)
        if p95 > rule.latency_budget_ms_p95:
            return PromotionDecision(promoted=False, reason=f"latency p95 {p95:.0f} ms > budget "
                                     f"{rule.latency_budget_ms_p95:.0f} ms", **decision)
        base_rate = None
        if base is not None and base.card.version != version:
            if rule.primary_engine not in base.eval_reports:
                base_rate = 0.0
            else:
                base_rate = self.report(base, rule.primary_engine).success_rate()
        decision["baseline_rate"] = base_rate
        if force:
            reason = "forced by user"
        elif base_rate is None:
            reason = "first production version" if base is None else "already production"
        elif ci[0] > base_rate:
            reason = f"CI low {ci[0]:.0%} > production {base.card.version} {base_rate:.0%}"
        else:
            return PromotionDecision(promoted=False, reason=f"does not beat production {base.card.version}: "
                                     f"CI low {ci[0]:.0%} <= {base_rate:.0%}", **decision)
        self._set_current(name, version, "promote", reason)
        return PromotionDecision(promoted=True, reason=reason, **decision)

    def rollback(self, name: str) -> RegisteredModel | None:
        """Make the previously promoted version production again; None (no production) if there is none."""
        prod = self._prod(name)
        stack: list[str] = []                       # replay history: promote pushes, rollback pops
        for h in prod["history"]:
            if h["action"] == "promote":
                stack.append(h["version"])
            elif stack:
                stack.pop()
        current = stack.pop() if stack else None
        previous = stack[-1] if stack else None
        self._set_current(name, previous, "rollback", f"rolled back from {current}")
        return self.get(name, previous) if previous else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m serve.registry", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "var/registry"))
    ap.add_argument("--rule", default=str(ROOT / "configs/serve.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("name")
    show = sub.add_parser("show")
    show.add_argument("name")
    show.add_argument("version")
    reg = sub.add_parser("register")
    reg.add_argument("--card", required=True, help="ModelCard YAML")
    reg.add_argument("--eval", action="append", default=[], help="engine=path/to/eval_report.json (repeatable)")
    pro = sub.add_parser("promote")
    pro.add_argument("name")
    pro.add_argument("version")
    pro.add_argument("--force", action="store_true")
    sub.add_parser("rollback").add_argument("name")
    sub.add_parser("resolve").add_argument("ref")
    a = ap.parse_args(argv)
    r = Registry(a.root)
    try:
        if a.cmd == "list":
            out = {"production": (r.production(a.name).card.version if r.production(a.name) else None),
                   "versions": [m.model_dump(mode="json") for m in r.list_versions(a.name)]}
        elif a.cmd == "show":
            out = r.get(a.name, a.version).model_dump(mode="json")
        elif a.cmd == "register":
            card = ModelCard.model_validate(yaml.safe_load(Path(a.card).read_text()))
            evals = dict(pair.split("=", 1) for pair in a.eval)
            out = r.register(card, evals).model_dump(mode="json")
        elif a.cmd == "promote":
            out = r.promote(a.name, a.version, PromotionRule.from_yaml(a.rule), force=a.force).model_dump(mode="json")
        elif a.cmd == "rollback":
            m = r.rollback(a.name)
            out = {"production": m.card.version if m else None}
        else:
            out = {"ref": a.ref, "policy": r.policy_for(a.ref)}
    except (KeyError, ValueError, FileExistsError, FileNotFoundError) as e:
        print(f"registry: {e}", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
