# Platform Plan B-loop: Versioning, Promotion, Failure Report, Targeted Next Run, ASSIST, Serving — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the sim loop in the Control Station without a GPU on the control machine:
1. evaluated checkpoints are **registered as versions**;
2. the best version is **promoted** (or rolled back) by a fixed rule;
3. rollout and eval can target a version, e.g. `registry:cathlab-pick@production`;
4. a **failure report** turns eval reports into failing cells and a `next_run.json`;
5. CREATE NEXT RUN starts a **targeted generate** in any engine;
6. the **ASSIST** panel turns a procedure step into a handover rollout of the production model;
7. a **policy server** serves a version over HTTP.

**Architecture:**
- **Pure-Python modules:**
  - `serve/registry.py`: versions, promotion, rollback; YAML under `var/registry/`.
  - `report/failure_report.py`: report and `next_run.json` from EvalReports.
  - `dataengine.variation.select_configs`: shared by every engine, with a new `cells:<next_run.json>` split.
- **Control plane:** new routers for registry, reports, next run and assist. The control plane resolves `registry:` policies to the concrete `lerobot:<ckpt>` or `expert` policy *before* dispatching to a worker, so engines never need the registry.
- **UI:** MODELS, REPORT and ASSIST tabs.
- **Serving:** `serve/{codec,policy_server,policy_client}.py` complete the `url:` policy path that `engines/policies.py` already supports.

**Spec:** [`docs/specs/2026-10-23-multi-engine-platform-design.md`](../specs/2026-10-23-multi-engine-platform-design.md) P9–P12. The exact contracts are adapted from [`docs/handoffs/2026-10-23-sharan-serve-intel-session.md`](../handoffs/2026-10-23-sharan-serve-intel-session.md) §3.1–3.4 and §3.6. Sharan is out; this plan replaces his S1–S3 and S6.

**Tech Stack:** Python ≥3.10, pydantic v2, PyYAML, numpy, FastAPI and httpx (control plane), stdlib `urllib` (policy client), vanilla ES modules, pytest.

## Global Constraints

- Commits are authored by the human only. **No `Co-authored-by:` trailers for AI agents.** Conventional Commit messages.
- Core tests (`uv run pytest -q`) stay GPU- and simulator-free; no network. Baseline before Task 1: **261 passed, 14 skipped**.
- Never write to `/tmp`; tests use `tmp_path`. Never edit `configs/variation/cathlab_v1.yaml`.
- Data, checkpoints, `.env` and secrets never go in git. The registry lives in `var/registry/` and reports in `var/reports/` (both git-ignored).
- Model name: `cathlab-pick`. Versions are `v1`, `v2`, …; references look like `cathlab-pick@v3` or `cathlab-pick@production`.
- **Promotion rule:** `primary_engine: robosuite`, `min_episodes: 30`, **`latency_budget_ms_p95: 500.0`** (user decision H31: VLA predict calls return action chunks; measured p95 SmolVLA 271 ms, X-VLA 214 ms, ACT 20 ms).
- Policy strings understood by every engine CLI: `expert`, `random`, `lerobot:<hf-repo-or-path>`, `url:<policy-server-url>`. The control plane alone also accepts `registry:<name>@<version|production>`.
- Code imported by engine envs (`dataengine/`, `serve/policy_client.py`, `serve/codec.py`) stays Python 3.10-compatible. `serve/policy_client.py` uses only stdlib and numpy.
- Worker and policy-server auth: header `X-Cathlab-Token`, env `CATHLAB_WORKER_TOKEN`. Ports: control 8700, policy server 8720.
- Frontend: vanilla ES modules, industrial PolyScope style (existing `station.css` classes). Every new JS file must pass `/home/ubuntu/miniforge3/envs/node/bin/node --input-type=module --check < file`.

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `serve/__init__.py`, `serve/registry.py`, `configs/serve.yaml` | Registry, PromotionRule/Decision, promote/rollback, CLI | 1 |
| `report/__init__.py`, `report/failure_report.py` | Cells, sim2sim gap, recommendations, `next_run.json`, CLI | 2 |
| `dataengine/variation.py` (+ `scripts/robosuite_generate.py`, `engines/maniskill/stages.py`, `engines/genesis/generate.py`) | Shared `select_configs` with `cells:` split; `cell_configs` | 3 |
| `orchestrator/loop.py`, `orchestrator/pipeline.py`, `orchestrator/control.py` | Eval discovery, registry, reports, next-run API; `registry:` policy resolution; targeted generate default | 4 |
| `frontend/js/models.js`, `frontend/js/report.js`, `frontend/index.html`, `frontend/js/app.js`, `frontend/css/station.css` | MODELS and REPORT tabs | 5 |
| `orchestrator/assist.py`, `frontend/js/assist.js` (+ html/app/css) | ASSIST tab and API | 6 |
| `serve/codec.py`, `serve/policy_server.py`, `serve/policy_client.py` | Serving (`url:` policies) | 7 |
| `docs/STATUS.md`, `docs/decisions.md`, `harness/gates/{rollout,report}.yaml`, `docs/EXTENDING.md`, `README.md`, `Makefile` | Docs, gate status, make targets | 8 |

---
### Task 1: Model registry with promotion rule and rollback

**Files:**
- Create: `serve/__init__.py`, `serve/registry.py`, `tests/serve/__init__.py`, `tests/serve/test_registry.py`
- Modify: `configs/serve.yaml`

**Interfaces:**
- Consumes:
  - `contracts.EvalReport` (`.load(path)`, `.success_rate()`, `.wilson_ci()`, `.episodes`), `contracts.ModelCard`;
  - `eval.metrics.latency_summary(report) -> {"p50_ms", "p95_ms", "max_ms"}`.
- Produces:
  - `serve.registry.MODEL_NAME = "cathlab-pick"`;
  - `RegisteredModel(card: ModelCard, eval_reports: dict[str, str], registered: float)`;
  - `PromotionRule(primary_engine="robosuite", latency_budget_ms_p95=500.0, min_episodes=30)` with `PromotionRule.from_yaml(path)`;
  - `PromotionDecision(promoted, reason, candidate_rate, candidate_ci, baseline_version, baseline_rate, p95_ms)`;
  - `Registry(root, repo_root=None)` with:
    - `.register(card, eval_reports)`;
    - `.list_versions(name)`, oldest first;
    - `.get(name, version)`, raising `KeyError`;
    - `.production(name)`;
    - `.history(name) -> list[dict]`;
    - `.promote(name, version, rule=None, force=False)`;
    - `.rollback(name)`;
    - `.resolve(ref) -> RegisteredModel`, where the ref is `"<name>@<version|production>"`;
    - `.policy_for(ref) -> str`.
  - `policy_string(card) -> str`: `expert` when `card.adapter == "expert"`, otherwise `lerobot:<card.ckpt_uri>`.
  - CLI: `python -m serve.registry [--root var/registry] {list|show|register|promote|rollback|resolve}`, printing JSON.
  - Relative eval-report paths are resolved against `repo_root` (default: the repo root).

- [ ] **Step 1: Write the failing tests** in `tests/serve/test_registry.py` (also create an empty `tests/serve/__init__.py`):

```python
import json
import pytest
from contracts import EpisodeResult, EvalReport, ModelCard
from serve.registry import MODEL_NAME, PromotionRule, Registry, main, policy_string


def report(tmp_path, name, successes, n=60, p95=100.0, engine="robosuite"):
    eps = [EpisodeResult(episode=i, seed=5000 + i % 4, object="vial", instruction="Hand me the vial",
                         success=i < successes, failed_stage=None if i < successes else "grasp", steps=100,
                         latency_ms_p50=p95 / 2, latency_ms_p95=p95, scene_variant="in_distribution")
           for i in range(n)]
    path = tmp_path / f"{name}_{engine}.json"
    path.write_text(EvalReport(model_version=name, dataset_versions=[], backend=engine, episodes=eps).model_dump_json())
    return str(path)


def card(version, ckpt="user/smolvla_panda_pick_v1", adapter="lerobot"):
    return ModelCard(model_name=MODEL_NAME, version=version, adapter=adapter, ckpt_uri=ckpt,
                     dataset_versions=["Rakeshdharan56/pick_robosuite@v1"], eval_report_path="")


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "registry", repo_root=tmp_path)


def test_register_list_get_and_duplicates(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 30)})
    assert [m.card.version for m in reg.list_versions(MODEL_NAME)] == ["v1", "v2"]
    assert reg.get(MODEL_NAME, "v2").card.eval_report_path.endswith("b_robosuite.json")
    with pytest.raises(FileExistsError):
        reg.register(card("v1"), {"robosuite": report(tmp_path, "c", 1)})
    with pytest.raises(KeyError):
        reg.get(MODEL_NAME, "v9")
    with pytest.raises(FileNotFoundError):
        reg.register(card("v3"), {"robosuite": str(tmp_path / "missing.json")})


def test_first_promotion_needs_only_latency_and_episodes(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20, p95=271.0)})
    d = reg.promote(MODEL_NAME, "v1")
    assert d.promoted and d.baseline_version is None and d.p95_ms == pytest.approx(271.0)
    assert reg.production(MODEL_NAME).card.version == "v1" and reg.get(MODEL_NAME, "v1").card.promoted


def test_candidate_must_beat_production_beyond_ci(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})       # 33 %
    reg.promote(MODEL_NAME, "v1")
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 24)})       # 40 %, CI low ~28 % < 33 %
    d = reg.promote(MODEL_NAME, "v2")
    assert not d.promoted and "does not beat" in d.reason and d.baseline_version == "v1"
    reg.register(card("v3"), {"robosuite": report(tmp_path, "c", 45)})       # 75 %, CI low ~63 % > 33 %
    assert reg.promote(MODEL_NAME, "v3").promoted
    assert reg.production(MODEL_NAME).card.version == "v3"
    assert not reg.get(MODEL_NAME, "v1").card.promoted and reg.get(MODEL_NAME, "v3").card.promoted


def test_latency_episodes_engine_and_force(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 50, p95=900.0)})
    d = reg.promote(MODEL_NAME, "v1")
    assert not d.promoted and "latency" in d.reason
    assert not reg.promote(MODEL_NAME, "v1", force=True).promoted          # force never skips latency
    reg.register(card("v2"), {"robosuite": report(tmp_path, "b", 5, n=10)})
    assert "insufficient eval" in reg.promote(MODEL_NAME, "v2").reason
    reg.register(card("v3"), {"maniskill": report(tmp_path, "c", 50, engine="maniskill")})
    assert "no eval on robosuite" in reg.promote(MODEL_NAME, "v3").reason
    reg.register(card("v4"), {"robosuite": report(tmp_path, "d", 50)})
    reg.promote(MODEL_NAME, "v4")
    reg.register(card("v5"), {"robosuite": report(tmp_path, "e", 1)})
    d = reg.promote(MODEL_NAME, "v5", force=True)
    assert d.promoted and d.reason == "forced by user"


def test_rollback_and_history(reg, tmp_path):
    for v, s in (("v1", 20), ("v2", 45)):
        reg.register(card(v), {"robosuite": report(tmp_path, v, s)})
        reg.promote(MODEL_NAME, v)
    assert reg.rollback(MODEL_NAME).card.version == "v1"
    assert reg.production(MODEL_NAME).card.version == "v1" and not reg.get(MODEL_NAME, "v2").card.promoted
    assert [h["action"] for h in reg.history(MODEL_NAME)] == ["promote", "promote", "rollback"]
    assert reg.rollback(MODEL_NAME) is None and reg.production(MODEL_NAME) is None


def test_resolve_and_policy_string(reg, tmp_path):
    reg.register(card("v1"), {"robosuite": report(tmp_path, "a", 20)})
    reg.register(card("v2", ckpt="", adapter="expert"), {"robosuite": report(tmp_path, "b", 60)})
    reg.promote(MODEL_NAME, "v1")
    assert reg.policy_for(f"{MODEL_NAME}@production") == "lerobot:user/smolvla_panda_pick_v1"
    assert reg.policy_for(f"{MODEL_NAME}@v2") == "expert" == policy_string(reg.get(MODEL_NAME, "v2").card)
    with pytest.raises(KeyError):
        reg.resolve("other@production")
    with pytest.raises(ValueError):
        reg.resolve("no-at-sign")


def test_rule_from_serve_yaml():
    rule = PromotionRule.from_yaml("configs/serve.yaml")
    assert rule.latency_budget_ms_p95 == 500.0 and rule.primary_engine == "robosuite" and rule.min_episodes == 30


def test_cli_register_promote_resolve(tmp_path, capsys):
    card_path = tmp_path / "card.yaml"
    card_path.write_text(f"model_name: {MODEL_NAME}\nversion: v1\nadapter: lerobot\nckpt_uri: user/m\n"
                         "dataset_versions: [d@v1]\neval_report_path: ''\n")
    root = ["--root", str(tmp_path / "reg")]
    assert main(root + ["register", "--card", str(card_path), "--eval", f"robosuite={report(tmp_path, 'a', 20)}"]) == 0
    assert main(root + ["promote", MODEL_NAME, "v1"]) == 0
    capsys.readouterr()
    assert main(root + ["resolve", f"{MODEL_NAME}@production"]) == 0
    assert json.loads(capsys.readouterr().out)["policy"] == "lerobot:user/m"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/serve/test_registry.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'serve'`.

- [ ] **Step 3: Implement.** `serve/__init__.py`:

```python
"""Serving: model registry (versions, promotion, rollback), policy server and client."""
```

`configs/serve.yaml` (replace the whole file):

```yaml
# Promotion rule and serving (spec P10). Latency budget per predict call (one action chunk); user decision H31.
latency_budget_ms_p95: 500.0
primary_engine: robosuite
min_episodes: 30
registry_path: var/registry
policy_server_port: 8720
```

`serve/registry.py`:

```python
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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/serve/test_registry.py` → 8 passed. Then `uv run pytest -q` → all pass.

- [ ] **Step 5: Smoke against the real eval reports** (the expected numbers come from `reports/robosuite/cross_physics_v1.md`):

```bash
R="--root var/smoke_registry"
printf 'model_name: cathlab-pick\nversion: v1\nadapter: lerobot\nckpt_uri: Rakeshdharan56/smolvla_panda_pick_v1\ndataset_versions: [Rakeshdharan56/pick_robosuite@v1]\neval_report_path: ""\n' > var/smoke_card.yaml
uv run python -m serve.registry $R register --card var/smoke_card.yaml \
  --eval robosuite=reports/robosuite/eval_reports/robosuite/smolvla_panda_pick_v1/eval_report.json \
  --eval maniskill=reports/robosuite/eval_reports/maniskill/smolvla_panda_pick_v1/eval_report.json
uv run python -m serve.registry $R promote cathlab-pick v1      # promoted: first production version, p95 ~271 ms
uv run python -m serve.registry $R resolve cathlab-pick@production
rm -rf var/smoke_registry var/smoke_card.yaml
```

- [ ] **Step 6: Commit**

```bash
git add serve configs/serve.yaml tests/serve
git commit -m "feat(serve): model registry with CI-based promotion rule, rollback and version resolution"
```

---
### Task 2: Failure report and `next_run.json`

**Files:**
- Create: `report/__init__.py`, `report/failure_report.py`, `tests/report/__init__.py`, `tests/report/test_failure_report.py`

**Interfaces:**
- Consumes:
  - `contracts.EvalReport`, `contracts.STAGES`;
  - `dataengine.variation`: `VariationSpec.load`, `eval_configs(spec, tier, [seed])`, `TIERS`;
  - `eval.metrics.wilson_ci(successes, n)`.
- Produces:
  - `report.failure_report.cell_of(spec, ep) -> dict | None`: `{tier, object, layout_id, obj_color_id, table_color_id}`, or None when `ep.scene_variant` is not a tier;
  - `build_report(evals: dict[str, EvalReport], spec, spec_path: str, model: str, top=10, k=8, primary="robosuite") -> dict`;
  - `render_markdown(report: dict) -> str`;
  - `next_run(report: dict) -> dict`;
  - `write_outputs(report, out_dir) -> dict[str, Path]`, which writes `report.json`, `report.md` and `next_run.json`;
  - CLI: `python -m report.failure_report --eval robosuite=<path> [--eval maniskill=<path>] --spec configs/variation/cathlab_v1.yaml --model <label> --out <dir> [--top 10] [--k 8] [--primary robosuite]`.
  - `next_run.json` = `{"spec": str, "model": str, "created": float, "k": int, "cells": [{"object", "layout_id", "obj_color_id", "table_color_id", "k"}]}`.
  - **Only cells whose layout, object colour and table colour are all in the spec's train splits enter `next_run.json`**: training on held-out values would leak the eval. Held-out failures get the "widen … in the next spec version" recommendation instead.

- [ ] **Step 1: Write the failing tests** in `tests/report/test_failure_report.py` (also create an empty `tests/report/__init__.py`):

```python
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
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/report/test_failure_report.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'report'`.

- [ ] **Step 3: Implement.** `report/__init__.py`:

```python
"""Failure report: eval reports -> failing cells, sim2sim gap, recommendations, next_run.json (closes the loop)."""
```

`report/failure_report.py`:

```python
"""Failure report from engine eval reports, and the targeted next run.

    python -m report.failure_report --eval robosuite=<eval_report.json> [--eval maniskill=...] \\
        --spec configs/variation/cathlab_v1.yaml --model cathlab-pick@v1 --out var/reports/<id> [--top 10] [--k 8]

A cell is (object, layout_id, obj_color_id, table_color_id), rebuilt from each episode's tier + seed with
dataengine.variation.eval_configs, so it is the same cell in every engine. next_run.json lists only cells whose
values are all in the spec's train splits: training on held-out values would leak the eval.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from contracts import STAGES, EvalReport
from dataengine.variation import TIERS, VariationSpec, eval_configs
from eval.metrics import wilson_ci

GENERATE_STAGES = ("reach", "grasp", "place")
EXPERT_STAGES = ("lift", "transport")
COLOUR_TIERS = ("held_out_obj_color", "held_out_table_color")


def cell_of(spec: VariationSpec, ep) -> dict | None:
    if ep.scene_variant not in TIERS:
        return None
    cfg = next((c for c in eval_configs(spec, ep.scene_variant, [ep.seed]) if c.object == ep.object), None)
    if cfg is None:
        return None
    return {"tier": ep.scene_variant, "object": ep.object, "layout_id": cfg.layout_id,
            "obj_color_id": cfg.obj_color_id, "table_color_id": cfg.table_color_id}


def _rate(successes: int, n: int) -> dict:
    lo, hi = wilson_ci(successes, n) if n else (0.0, 0.0)
    return {"successes": successes, "n": n, "rate": round(successes / n, 4) if n else 0.0,
            "ci": [round(lo, 4), round(hi, 4)]}


def _in_train(spec: VariationSpec, cell: dict) -> bool:
    return (cell["layout_id"] in spec.layout_split.train and cell["obj_color_id"] in spec.obj_color_split.train
            and cell["table_color_id"] in spec.table_color_split.train)


def _recommend(stage: str, tiers: list[str], k: int, trainable: bool) -> str:
    if not trainable:
        if set(tiers) <= set(COLOUR_TIERS):
            return "widen the colour palette (object/table colours) in the next spec version"
        return "widen the layout/colour ranges in the next spec version (held-out values are never trained on)"
    if stage in EXPERT_STAGES:
        return "check expert params / friction range, then generate targeted episodes"
    return f"generate {k} targeted episodes"


def build_report(evals: dict[str, EvalReport], spec: VariationSpec, spec_path: str, model: str, top: int = 10,
                 k: int = 8, primary: str = "robosuite") -> dict:
    overall, by_cell, tier_rates, stages, wrong = {}, [], defaultdict(dict), {}, {}
    cells: dict[tuple, dict] = {}
    for engine, rep in evals.items():
        eps = rep.episodes
        overall[engine] = _rate(sum(e.success for e in eps), len(eps))
        stages[engine] = {s: sum(1 for e in eps if not e.success and e.failed_stage == s) for s in STAGES}
        wrong[engine] = sum(1 for e in eps if e.judge_verdict and "wrong" in e.judge_verdict.lower())
        groups = defaultdict(list)
        for e in eps:
            groups[(e.scene_variant, e.object)].append(e)
        for (tier, obj), g in sorted(groups.items()):
            by_cell.append({"engine": engine, "tier": tier, "object": obj, **_rate(sum(e.success for e in g), len(g))})
        for tier in TIERS:
            g = [e for e in eps if e.scene_variant == tier]
            if g:
                tier_rates[tier][engine] = sum(e.success for e in g) / len(g)
        for e in eps:
            if e.success:
                continue
            cell = cell_of(spec, e)
            if cell is None:
                continue
            key = (cell["object"], cell["layout_id"], cell["obj_color_id"], cell["table_color_id"])
            c = cells.setdefault(key, {"object": key[0], "layout_id": key[1], "obj_color_id": key[2],
                                       "table_color_id": key[3], "failures": 0, "engines": Counter(),
                                       "stages": Counter(), "tiers": [], "video": None})
            c["failures"] += 1
            c["engines"][engine] += 1
            c["stages"][e.failed_stage or "reach"] += 1
            if cell["tier"] not in c["tiers"]:
                c["tiers"].append(cell["tier"])
            c["video"] = c["video"] or e.video_path
    failing = sorted(cells.values(), key=lambda c: (-c["failures"], c["object"], c["layout_id"],
                                                    c["obj_color_id"], c["table_color_id"]))[:top]
    for c in failing:
        stage = c["stages"].most_common(1)[0][0]
        c["trainable"] = _in_train(spec, c)
        c["main_stage"] = stage
        c["recommendation"] = _recommend(stage, c["tiers"], k, c["trainable"])
        c["engines"], c["stages"] = dict(c["engines"]), dict(c["stages"])
    gap = []
    for tier in TIERS:
        rates = tier_rates.get(tier, {})
        if primary in rates and len(rates) > 1:
            gap.append({"tier": tier, "primary": round(rates[primary], 4),
                        "gaps": {e: round(rates[primary] - r, 4) for e, r in rates.items() if e != primary}})
    return {"model": model, "spec": spec_path, "primary": primary, "k": k, "created": time.time(),
            "overall": overall, "by_tier_object": by_cell, "stage_failures": stages, "wrong_object": wrong,
            "sim2sim_gap": gap, "failing_cells": failing}


def next_run(report: dict) -> dict:
    cells = [{"object": c["object"], "layout_id": c["layout_id"], "obj_color_id": c["obj_color_id"],
              "table_color_id": c["table_color_id"], "k": report["k"]}
             for c in report["failing_cells"] if c["trainable"]]
    return {"spec": report["spec"], "model": report["model"], "created": report["created"], "k": report["k"],
            "cells": cells}


def render_markdown(r: dict) -> str:
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    lines = [f"# Failure report: {r['model']}", "", f"Spec `{r['spec']}`, primary engine `{r['primary']}`.", "",
             "## Overall", "", "| Engine | Success | 95 % CI | n |", "|---|---|---|---|"]
    for e, o in r["overall"].items():
        lines.append(f"| {e} | {pct(o['rate'])} | {pct(o['ci'][0])}–{pct(o['ci'][1])} | {o['n']} |")
    lines += ["", "## Failure stage", "", "| Engine | " + " | ".join(STAGES) + " | wrong object |",
              "|---" * (len(STAGES) + 2) + "|"]
    for e, h in r["stage_failures"].items():
        lines.append(f"| {e} | " + " | ".join(str(h[s]) for s in STAGES) + f" | {r['wrong_object'][e]} |")
    lines += ["", "## Sim2sim gap", "", f"Primary ({r['primary']}) minus each other engine, per tier.", ""]
    if r["sim2sim_gap"]:
        lines += ["| Tier | " + r["primary"] + " | gaps |", "|---|---|---|"]
        for g in r["sim2sim_gap"]:
            gaps = ", ".join(f"{e}: {v:+.0%}" for e, v in g["gaps"].items())
            lines.append(f"| {g['tier']} | {pct(g['primary'])} | {gaps} |")
    else:
        lines.append("Only one engine evaluated.")
    lines += ["", "## Top failing cells", "",
              "| Object | Layout | Obj colour | Table colour | Failures | Main stage | Tiers | Recommendation | Video |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in r["failing_cells"]:
        lines.append(f"| {c['object']} | {c['layout_id']} | {c['obj_color_id']} | {c['table_color_id']} | "
                     f"{c['failures']} | {c['main_stage']} | {', '.join(c['tiers'])} | {c['recommendation']} | "
                     f"{c['video'] or ''} |")
    nr = next_run(r)
    lines += ["", "## Next run", "",
              f"`next_run.json`: {len(nr['cells'])} train-split cells × {r['k']} episodes = "
              f"{len(nr['cells']) * r['k']} targeted episodes. Held-out cells are never trained on.", ""]
    return "\n".join(lines)


def write_outputs(report: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report.json": out / "report.json", "report.md": out / "report.md", "next_run.json": out / "next_run.json"}
    paths["report.json"].write_text(json.dumps(report, indent=2))
    paths["report.md"].write_text(render_markdown(report))
    paths["next_run.json"].write_text(json.dumps(next_run(report), indent=2))
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m report.failure_report", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True, help="engine=path/to/eval_report.json (repeatable)")
    ap.add_argument("--spec", default="configs/variation/cathlab_v1.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--primary", default="robosuite")
    a = ap.parse_args(argv)
    try:
        evals = {e: EvalReport.load(p) for e, p in (pair.split("=", 1) for pair in a.eval)}
    except (OSError, ValueError) as e:
        print(f"failure_report: {e}", file=sys.stderr)
        return 1
    report = build_report(evals, VariationSpec.load(a.spec), a.spec, a.model, a.top, a.k, a.primary)
    paths = write_outputs(report, a.out)
    print(f"wrote {', '.join(str(p) for p in paths.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/report` → 3 passed. Then `uv run pytest -q` → all pass.

- [ ] **Step 5: Smoke on the real reports**

```bash
uv run python -m report.failure_report --model smolvla_panda_pick_v1 --out var/smoke_report \
  --eval robosuite=reports/robosuite/eval_reports/robosuite/smolvla_panda_pick_v1/eval_report.json \
  --eval maniskill=reports/robosuite/eval_reports/maniskill/smolvla_panda_pick_v1/eval_report.json
head -40 var/smoke_report/report.md; python -c "import json; print(len(json.load(open('var/smoke_report/next_run.json'))['cells']))"
rm -rf var/smoke_report
```

- [ ] **Step 6: Commit**

```bash
git add report tests/report
git commit -m "feat(report): failure report with failing cells, sim2sim gap, recommendations and next_run.json"
```

---
### Task 3: Shared config selection with a targeted `cells:` split (every engine)

**Files:**
- Modify:
  - `dataengine/variation.py` (add `TARGETED_SEED0`, `cell_configs`, `load_cells`, `select_configs`);
  - `scripts/robosuite_generate.py`, `engines/maniskill/stages.py`, `engines/genesis/generate.py`: replace each local `select_configs` body with the shared one, keeping the name importable from each module.
- Create: `tests/dataengine/test_targeted.py`

**Interfaces:**
- Consumes: `next_run.json` from Task 2: `{"spec", "cells": [{"object", "layout_id", "obj_color_id", "table_color_id", "k"}], ...}`.
- Produces:
  - `dataengine.variation.TARGETED_SEED0 = 200_000`;
  - `load_cells(path) -> list[dict]`;
  - `cell_configs(spec, cells) -> list[EpisodeConfig]`: `k` configs per cell, values fixed; mass, friction and light are sampled from `default_rng(seed)` with `seed = TARGETED_SEED0 + 1000 * cell_index + j`; `tier = "targeted"`. Raises `ValueError` for an unknown object or an out-of-range id.
  - `select_configs(spec, split, start, n, shard, num_shards, eval_seed0=100_000) -> list[EpisodeConfig]`. `split` is `train`, a tier, or `cells:<path to next_run.json>`. All three use `generation_order(...)[start:start + n][shard::num_shards]`, so a targeted run is object-balanced and shardable. Unknown split → `ValueError`.
  - Every engine's `generate --split cells:<abs path>` therefore works unchanged (the flag already passes through).

- [ ] **Step 1: Write the failing tests** in `tests/dataengine/test_targeted.py`:

```python
import json
import pytest
from dataengine.variation import (TARGETED_SEED0, VariationSpec, cell_configs, generation_order, select_configs,
                                  train_configs)

SPEC = VariationSpec.load("configs/variation/cathlab_v1.yaml")
CELLS = [{"object": "bottle", "layout_id": 0, "obj_color_id": 0, "table_color_id": 0, "k": 3},
         {"object": "vial", "layout_id": 2, "obj_color_id": 2, "table_color_id": 2, "k": 2}]


def test_cell_configs_fix_the_cell_and_vary_the_rest():
    cfgs = cell_configs(SPEC, CELLS)
    assert len(cfgs) == 5 and all(c.tier == "targeted" for c in cfgs)
    bottles = [c for c in cfgs if c.object == "bottle"]
    assert {(c.layout_id, c.obj_color_id, c.table_color_id) for c in bottles} == {(0, 0, 0)}
    assert [c.seed for c in bottles] == [TARGETED_SEED0, TARGETED_SEED0 + 1, TARGETED_SEED0 + 2]
    assert len({c.mass_scale for c in bottles}) == 3
    assert cell_configs(SPEC, CELLS) == cfgs                                  # deterministic


@pytest.mark.parametrize("bad", [{"object": "scalpel"}, {"layout_id": 99}, {"obj_color_id": -1}, {"table_color_id": 50}])
def test_cell_configs_validate(bad):
    with pytest.raises(ValueError):
        cell_configs(SPEC, [{**CELLS[0], **bad}])


def test_select_configs_cells_split_and_shards(tmp_path):
    p = tmp_path / "next_run.json"
    p.write_text(json.dumps({"spec": "x", "cells": CELLS}))
    full = select_configs(SPEC, f"cells:{p}", 0, 5, 0, 1)
    assert full == generation_order(SPEC, cell_configs(SPEC, CELLS))
    assert [c.object for c in full[:2]] == ["bottle", "vial"]                 # objects interleaved
    shards = [select_configs(SPEC, f"cells:{p}", 0, 5, i, 2) for i in range(2)]
    assert sorted(c.seed for s in shards for c in s) == sorted(c.seed for c in full)
    assert select_configs(SPEC, f"cells:{p}", 0, 2, 0, 1) == full[:2]


def test_select_configs_train_matches_old_window_and_rejects_unknown():
    assert select_configs(SPEC, "train", 10, 6, 1, 2) == generation_order(SPEC, train_configs(SPEC))[10:16][1::2]
    with pytest.raises(ValueError, match="split"):
        select_configs(SPEC, "bogus", 0, 1, 0, 1)


def test_engines_share_the_function():
    import importlib
    from dataengine import variation
    for mod in ("engines.maniskill.stages", "engines.genesis.generate"):
        assert importlib.import_module(mod).select_configs is variation.select_configs
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/dataengine/test_targeted.py`
Expected: FAIL with `ImportError: cannot import name 'TARGETED_SEED0'`.

- [ ] **Step 3: Implement.** In `dataengine/variation.py`, add `import json` to the imports and append:

```python
TARGETED_SEED0 = 200_000   # targeted (next-run) episodes; never overlaps train (< 1e6 * 0 + grid) or eval (>= 5000) seeds


def load_cells(path: str | Path) -> list[dict]:
    """Cells of a next_run.json (report/failure_report.py)."""
    return json.loads(Path(path).read_text())["cells"]


def cell_configs(spec: VariationSpec, cells: list[dict]) -> list[EpisodeConfig]:
    """k configs per failing cell: object, layout and colours fixed; mass, friction and light resampled."""
    out = []
    for ci, cell in enumerate(cells):
        if cell["object"] not in spec.objects:
            raise ValueError(f"unknown object {cell['object']!r} in cell {ci}")
        for key, size in (("layout_id", spec.n_layouts), ("obj_color_id", len(spec.obj_colors)),
                          ("table_color_id", len(spec.table_colors))):
            if not 0 <= int(cell[key]) < size:
                raise ValueError(f"cell {ci}: {key}={cell[key]} outside 0..{size - 1}")
        for j in range(int(cell.get("k", 8))):
            seed = TARGETED_SEED0 + 1000 * ci + j
            rng = np.random.default_rng(seed)
            out.append(EpisodeConfig(object=cell["object"], layout_id=int(cell["layout_id"]),
                                     obj_color_id=int(cell["obj_color_id"]), table_color_id=int(cell["table_color_id"]),
                                     seed=seed, tier="targeted", **_continuous(spec, rng)))
    return out


def select_configs(spec: VariationSpec, split: str, start: int, n: int, shard: int, num_shards: int,
                   eval_seed0: int = 100_000) -> list[EpisodeConfig]:
    """The configs a generate run works on, the same in every engine (episode i is paired across engines).
    split: 'train', an eval tier, or 'cells:<next_run.json>' (targeted next run)."""
    if split == "train":
        cfgs = train_configs(spec)
    elif split in TIERS:
        cfgs = eval_configs(spec, split, list(range(eval_seed0, eval_seed0 + (start + n + 2) // 3 + 1)))
    elif split.startswith("cells:"):
        cfgs = cell_configs(spec, load_cells(split[len("cells:"):]))
    else:
        raise ValueError(f"--split must be 'train', one of {TIERS} or 'cells:<next_run.json>', got {split!r}")
    return generation_order(spec, cfgs)[start:start + n][shard::num_shards]
```

In each of `scripts/robosuite_generate.py`, `engines/maniskill/stages.py` and `engines/genesis/generate.py`:
- delete the local `def select_configs(...)` function;
- import it instead: `from dataengine.variation import select_configs`, added to the existing `dataengine.variation` import;
- remove any of `TIERS`, `eval_configs`, `generation_order` and `train_configs` that become unused. Keep any that are still used elsewhere in the file (e.g. `check_configs` in maniskill).

The call sites stay the same. Update each module's `--split` help text to "'train', an eval tier, or cells:<next_run.json>".

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/dataengine tests/engines` → all pass, including the existing pairing tests in `test_maniskill_cli.py` and `test_genesis_cli.py`. Then `uv run pytest -q`.

- [ ] **Step 5: Real smoke in robosuite** (2 targeted episodes):

```bash
printf '{"spec": "configs/variation/cathlab_v1.yaml", "cells": [{"object": "vial", "layout_id": 1, "obj_color_id": 1, "table_color_id": 1, "k": 2}]}' > var/smoke_next_run.json
MUJOCO_GL=egl conda run -n robosuite --no-capture-output python -m engines.robosuite.cli generate \
  --out var/smoke_targeted --split cells:$PWD/var/smoke_next_run.json --n 2
ls var/smoke_targeted/raw/*/episodes | head; rm -rf var/smoke_targeted var/smoke_next_run.json
```

Expected: 2 episodes kept (the expert is 100% on train cells).

- [ ] **Step 6: Commit**

```bash
git add dataengine/variation.py scripts/robosuite_generate.py engines/maniskill/stages.py engines/genesis/generate.py tests/dataengine/test_targeted.py
git commit -m "feat(dataengine): shared select_configs with a targeted cells:<next_run.json> split for every engine"
```

---
### Task 4: Control-plane loop API (evals, registry, reports, next run) and `registry:` policies in the pipeline

**Files:**
- Create: `orchestrator/loop.py`, `tests/orchestrator/test_loop.py`
- Modify: `orchestrator/pipeline.py` (`resolve_policy`, `run_defaults`, `effective_params`), `orchestrator/control.py` (registry + router wiring, `eval_roots` param)

**Interfaces:**
- Consumes:
  - from Task 1: `Registry(root, repo_root)` with `.register`, `.list_versions`, `.production`, `.history`, `.promote`, `.rollback`, `.abs_path`, `.policy_for(ref)`; also `PromotionRule.from_yaml`, `MODEL_NAME`;
  - from Task 2: `build_report`, `write_outputs`, `render_markdown`, `next_run`;
  - from Task 3: `dataengine.variation.load_cells`;
  - `orchestrator.catalog.load_catalog`, `selection_problems`; `Store.create_run`; `new_id`.
- Produces:
  - HTTP under `/api`:
    - `GET /evals` → `[{path, engine, model, n, rate, ci, p95_ms}]`: every `eval_report.json` under the eval roots (default: `<repo>/reports` and `<var>/runs`);
    - `GET /registry` → `{name, production, rule, versions: [RegisteredModel + summary: {engine: {n, rate, ci, p95_ms}}], history}`;
    - `POST /registry/register` with `{version?, adapter: "lerobot"|"expert", ckpt_uri, dataset_versions, eval_reports: {engine: path}, wandb_run?}` → the version view. Without a version it gets the next `vN`. Errors: 409 if the version exists; 400 on a bad path, adapter or version.
    - `POST /registry/promote {version, force=false}` → `PromotionDecision` (404 on an unknown version);
    - `POST /registry/rollback` → `{production}`;
    - `POST /reports {eval_reports: {engine: path}, model, k=8, top=10, spec="configs/variation/cathlab_v1.yaml"}` → `{id, report, markdown, next_run}`, written to `<var>/reports/<id>/`;
    - `GET /reports` → `[{id, model, created, overall, cells}]`, newest first;
    - `GET /reports/{id}` → `{id, report, markdown, next_run}`;
    - `POST /reports/{id}/next-run {selection?, note?}` → a new Run whose selection carries `next_run: <abs path>` (409 if it has no train cells or the selection is not runnable).
  - Eval report paths in requests must resolve to an existing `.json` under the repo or `<var>`; otherwise 400.
  - `Pipeline(..., resolve_policy: Callable[[str], str] | None = None)`:
    - `run_defaults(run, stage) -> dict`: for `generate` in a next run, `{split: "cells:<next_run>", start: 0, n: <sum of k>}`;
    - `effective_params(run, stage, params) -> dict`: run defaults merged under the user params. A `policy` of `registry:<ref>` is resolved to the engine policy, and for eval `model-version` defaults to `<ref>`. Raises `PipelineError(400)` on an unknown ref.
    - `stages(run)` returns param defaults with the run defaults applied, so the UI shows the targeted split.
  - `create_control_app(..., eval_roots: list | None = None)`.

- [ ] **Step 1: Write the failing tests** in `tests/orchestrator/test_loop.py`:

```python
import json
import pytest
from fastapi.testclient import TestClient
from contracts import EpisodeResult, EvalReport
from orchestrator.control import create_control_app
from orchestrator.pipeline import PipelineError
from orchestrator.stages import STAGE_BY_ID
from orchestrator.worker import create_worker_app
from tests.orchestrator.test_pipeline import CATALOG, ROOT, router, wait_done


def write_report(var, engine, model, successes, n=60, p95=100.0, tier="in_distribution"):
    eps = [EpisodeResult(episode=i, seed=5000 + i % 4, object=["bottle", "vial", "syringe"][i % 3],
                         instruction="x", success=i < successes, failed_stage=None if i < successes else "grasp",
                         steps=50, latency_ms_p50=p95 / 2, latency_ms_p95=p95, scene_variant=tier) for i in range(n)]
    p = var / "evals" / engine / model / "eval_report.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(EvalReport(model_version=model, dataset_versions=[], backend=engine, episodes=eps).model_dump_json())
    return str(p)


@pytest.fixture
def env(tmp_path):
    var = tmp_path / "var"
    (tmp_path / "catalog.yaml").write_text(CATALOG)
    (tmp_path / "workers.yaml").write_text('workers:\n  - {name: fake, url: "http://fake-worker"}\n')
    worker = TestClient(create_worker_app("fake", ["convert"], tmp_path / "worker"))
    app = create_control_app(ROOT, var_dir=var, catalog_path=tmp_path / "catalog.yaml",
                             workers_path=tmp_path / "workers.yaml", frontend_dir=tmp_path / "none",
                             engines_dir=ROOT / "tests/engines", worker_http=router({"http://fake-worker": worker}),
                             eval_roots=[var / "evals"])
    return TestClient(app), var, app


def test_evals_are_discovered(env):
    c, var, _ = env
    write_report(var, "robosuite", "smolvla", 23, p95=271.0)
    write_report(var, "maniskill", "smolvla", 0)
    evals = {(e["engine"], e["model"]): e for e in c.get("/api/evals").json()}
    assert evals[("robosuite", "smolvla")]["rate"] == round(23 / 60, 4)
    assert evals[("robosuite", "smolvla")]["p95_ms"] == pytest.approx(271.0)
    assert ("maniskill", "smolvla") in evals


def test_register_promote_rollback(env, tmp_path):
    c, var, _ = env
    a, b = write_report(var, "robosuite", "a", 20), write_report(var, "robosuite", "b", 24)
    v1 = c.post("/api/registry/register", json={"ckpt_uri": "u/a", "eval_reports": {"robosuite": a}}).json()
    assert v1["card"]["version"] == "v1" and v1["summary"]["robosuite"]["n"] == 60
    assert not v1["eval_reports"]["robosuite"].startswith("/") or "var" in v1["eval_reports"]["robosuite"]
    assert c.post("/api/registry/register", json={"ckpt_uri": "u/b", "eval_reports": {"robosuite": b}}).json()["card"]["version"] == "v2"
    assert c.post("/api/registry/register", json={"version": "v2", "ckpt_uri": "u", "eval_reports": {"robosuite": b}}).status_code == 409
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    assert c.post("/api/registry/register", json={"ckpt_uri": "u", "eval_reports": {"robosuite": str(outside)}}).status_code == 400
    assert c.post("/api/registry/register", json={"adapter": "lerobot", "ckpt_uri": "", "eval_reports": {"robosuite": a}}).status_code == 400
    assert c.post("/api/registry/promote", json={"version": "v1"}).json()["promoted"]
    d = c.post("/api/registry/promote", json={"version": "v2"}).json()
    assert not d["promoted"] and "does not beat" in d["reason"]
    assert c.post("/api/registry/promote", json={"version": "v9"}).status_code == 404
    reg = c.get("/api/registry").json()
    assert reg["production"] == "v1" and [v["card"]["version"] for v in reg["versions"]] == ["v1", "v2"]
    assert reg["rule"]["latency_budget_ms_p95"] == 500.0
    assert c.post("/api/registry/rollback").json() == {"production": None}


def test_report_and_next_run(env):
    c, var, _ = env
    rs = write_report(var, "robosuite", "m", 10)
    r = c.post("/api/reports", json={"eval_reports": {"robosuite": rs}, "model": "m"}).json()
    assert r["markdown"].startswith("# Failure report: m") and r["next_run"]["cells"]
    assert c.get("/api/reports").json()[0]["id"] == r["id"]
    assert c.get(f"/api/reports/{r['id']}").json()["report"]["model"] == "m"
    assert c.get("/api/reports/..%2F..%2Fetc").status_code in (400, 404)
    run = c.post(f"/api/reports/{r['id']}/next-run", json={"note": "loop 1"}).json()
    assert run["selection"]["next_run"].endswith(f"{r['id']}/next_run.json") and run["note"] == "loop 1"
    total = sum(cell["k"] for cell in r["next_run"]["cells"])
    gen = {s["id"]: s for s in c.get(f"/api/runs/{run['id']}/stages").json()}["generate"]
    defaults = {p["name"]: p["default"] for p in gen["params"]}
    assert defaults["split"] == f"cells:{run['selection']['next_run']}" and defaults["n"] == total
    job = c.post(f"/api/runs/{run['id']}/stages/generate", json={"params": {}}).json()
    assert wait_done(c, job["id"])[0] == "done"
    argv = c.get(f"/api/jobs/{job['id']}").json()["result"]["argv"]
    assert argv[argv.index("--split") + 1].startswith("cells:") and argv[argv.index("--n") + 1] == str(total)


def test_registry_policies_resolve_in_the_pipeline(env):
    c, var, app = env
    a = write_report(var, "robosuite", "x", 50)
    c.post("/api/registry/register", json={"adapter": "expert", "eval_reports": {"robosuite": a}})
    run = c.post("/api/runs", json={"selection": {"simulator": "fake", "expert": "analytic", "objects": ["vial"]}}).json()
    pipe, r = app.state.pipeline, app.state.store.get_run(run["id"])
    with pytest.raises(PipelineError) as e:
        pipe.effective_params(r, STAGE_BY_ID["eval"], {"policy": "registry:cathlab-pick@production"})
    assert e.value.status == 400
    c.post("/api/registry/promote", json={"version": "v1"})
    p = pipe.effective_params(r, STAGE_BY_ID["eval"], {"policy": "registry:cathlab-pick@production"})
    assert p["policy"] == "expert" and p["model-version"] == "cathlab-pick@production"
    assert pipe.effective_params(r, STAGE_BY_ID["rollout"], {"policy": "registry:cathlab-pick@v1"})["policy"] == "expert"
    assert pipe.effective_params(r, STAGE_BY_ID["generate"], {"n": 3}) == {"n": 3}
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest -q tests/orchestrator/test_loop.py`
Expected: FAIL with `TypeError: create_control_app() got an unexpected keyword argument 'eval_roots'`.

- [ ] **Step 3: Implement `orchestrator/loop.py`:**

```python
"""Closed-loop API of the control plane: eval reports, model registry (versions, promote, rollback), failure reports
and the targeted next run. Mounted under /api by orchestrator/control.py."""
import json
import re
import subprocess
from pathlib import Path
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from contracts import EvalReport, ModelCard
from dataengine.variation import VariationSpec
from eval.metrics import latency_summary
from orchestrator.catalog import load_catalog, selection_problems
from orchestrator.store import Store, new_id
from report.failure_report import build_report, write_outputs
from serve.registry import MODEL_NAME, PromotionRule, Registry

VERSION_RE = re.compile(r"^v(\d+)$")
REPORT_ID_RE = re.compile(r"^rep-[A-Za-z0-9-]+$")


class RegisterBody(BaseModel):
    version: str | None = None
    adapter: str = "lerobot"
    ckpt_uri: str = ""
    dataset_versions: list[str] = Field(default_factory=list)
    eval_reports: dict[str, str]
    wandb_run: str | None = None


class PromoteBody(BaseModel):
    version: str
    force: bool = False


class ReportBody(BaseModel):
    eval_reports: dict[str, str]
    model: str
    k: int = Field(8, ge=1, le=100)
    top: int = Field(10, ge=1, le=100)
    spec: str = "configs/variation/cathlab_v1.yaml"


class NextRunBody(BaseModel):
    selection: dict | None = None
    note: str = ""


def summarize(path: Path) -> dict:
    r = EvalReport.load(path)
    lo, hi = r.wilson_ci()
    return {"engine": r.backend, "model": r.model_version, "n": len(r.episodes), "rate": round(r.success_rate(), 4),
            "ci": [round(lo, 4), round(hi, 4)], "p95_ms": round(latency_summary(r)["p95_ms"], 1)}


def create_loop_router(store: Store, registry: Registry, var_dir: Path, repo_root: Path, catalog_path: Path,
                       eval_roots: list[Path]) -> APIRouter:
    r = APIRouter()
    var_dir, repo_root = Path(var_dir), Path(repo_root)
    reports_dir = var_dir / "reports"
    serve_yaml = repo_root / "configs/serve.yaml"
    rule = PromotionRule.from_yaml(serve_yaml) if serve_yaml.exists() else PromotionRule()
    allowed = [repo_root.resolve(), var_dir.resolve()]

    def under_allowed(p: Path) -> bool:
        return any(a == p or a in p.parents for a in allowed)

    def checked(path: str, suffix: str = ".json") -> Path:
        p = Path(path)
        p = (p if p.is_absolute() else repo_root / p).resolve()
        if p.suffix != suffix or not under_allowed(p) or not p.is_file():
            raise HTTPException(400, f"not a {suffix} file under the repo or var/: {path}")
        return p

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            return str(p.resolve())

    def safe_summary(path: Path) -> dict | None:
        try:
            return summarize(path)
        except (OSError, ValueError):
            return None

    def git_sha() -> str | None:
        try:
            return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, capture_output=True,
                                  text=True, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    def version_view(m) -> dict:
        return {**m.model_dump(mode="json"),
                "summary": {e: safe_summary(registry.abs_path(p)) for e, p in m.eval_reports.items()}}

    # evals -----------------------------------------------------------------------------------------
    @r.get("/evals")
    def evals() -> list[dict]:
        out = []
        for root in eval_roots:
            root = Path(root)
            if not root.exists():
                continue
            for p in sorted(root.glob("**/eval_report.json")):
                s = safe_summary(p)
                if s is not None:
                    out.append({"path": rel(p), **s})
        return out

    # registry --------------------------------------------------------------------------------------
    @r.get("/registry")
    def get_registry() -> dict:
        prod = registry.production(MODEL_NAME)
        return {"name": MODEL_NAME, "production": prod.card.version if prod else None, "rule": rule.model_dump(),
                "versions": [version_view(m) for m in registry.list_versions(MODEL_NAME)],
                "history": registry.history(MODEL_NAME)}

    @r.post("/registry/register")
    def register(body: RegisterBody) -> dict:
        if body.adapter not in ("lerobot", "expert"):
            raise HTTPException(400, "adapter must be 'lerobot' or 'expert'")
        if body.adapter == "lerobot" and not body.ckpt_uri.strip():
            raise HTTPException(400, "ckpt_uri is required for a lerobot model (HF repo id or path)")
        if not body.eval_reports:
            raise HTTPException(400, "at least one eval report is required")
        evals_ = {engine: rel(checked(p)) for engine, p in body.eval_reports.items()}
        existing = [int(m.group(1)) for v in registry.list_versions(MODEL_NAME)
                    if (m := VERSION_RE.match(v.card.version))]
        version = body.version or f"v{max(existing, default=0) + 1}"
        if not VERSION_RE.match(version):
            raise HTTPException(400, "version must look like v<N>")
        card = ModelCard(model_name=MODEL_NAME, version=version, adapter=body.adapter, ckpt_uri=body.ckpt_uri.strip(),
                         dataset_versions=body.dataset_versions, eval_report_path="", wandb_run=body.wandb_run,
                         git_sha=git_sha())
        try:
            return version_view(registry.register(card, evals_))
        except FileExistsError as e:
            raise HTTPException(409, str(e)) from e

    @r.post("/registry/promote")
    def promote(body: PromoteBody) -> dict:
        try:
            return registry.promote(MODEL_NAME, body.version, rule, force=body.force).model_dump(mode="json")
        except KeyError as e:
            raise HTTPException(404, str(e)) from e

    @r.post("/registry/rollback")
    def rollback() -> dict:
        m = registry.rollback(MODEL_NAME)
        return {"production": m.card.version if m else None}

    # failure reports + next run ----------------------------------------------------------------
    def report_dir(rid: str) -> Path:
        if not REPORT_ID_RE.match(rid) or not (reports_dir / rid / "report.json").is_file():
            raise HTTPException(404, f"unknown report {rid}")
        return reports_dir / rid

    def load_report(rid: str) -> dict:
        d = report_dir(rid)
        return {"id": rid, "report": json.loads((d / "report.json").read_text()),
                "markdown": (d / "report.md").read_text(), "next_run": json.loads((d / "next_run.json").read_text())}

    @r.post("/reports")
    def create_report(body: ReportBody) -> dict:
        evals_ = {engine: checked(p) for engine, p in body.eval_reports.items()}
        if not evals_:
            raise HTTPException(400, "at least one eval report is required")
        spec_path = checked(body.spec, ".yaml")
        try:
            reports = {e: EvalReport.load(p) for e, p in evals_.items()}
        except ValueError as e:
            raise HTTPException(400, f"invalid eval report: {e}") from e
        report = build_report(reports, VariationSpec.load(spec_path), rel(spec_path), body.model, body.top, body.k)
        report["eval_reports"] = {e: rel(p) for e, p in evals_.items()}
        rid = new_id("rep")
        write_outputs(report, reports_dir / rid)
        return load_report(rid)

    @r.get("/reports")
    def list_reports() -> list[dict]:
        out = []
        for d in reports_dir.glob("rep-*/report.json") if reports_dir.exists() else []:
            rep = json.loads(d.read_text())
            nr = json.loads((d.parent / "next_run.json").read_text())
            out.append({"id": d.parent.name, "model": rep["model"], "created": rep["created"],
                        "overall": rep["overall"], "cells": len(nr["cells"])})
        return sorted(out, key=lambda x: -x["created"])

    @r.get("/reports/{rid}")
    def get_report(rid: str) -> dict:
        return load_report(rid)

    @r.post("/reports/{rid}/next-run")
    def create_next_run(rid: str, body: NextRunBody) -> dict:
        path = (report_dir(rid) / "next_run.json").resolve()
        nr = json.loads(path.read_text())
        if not nr["cells"]:
            raise HTTPException(409, "the report has no train-split failing cells to target")
        catalog = load_catalog(catalog_path)
        selection = body.selection or dict(catalog.defaults)
        problems = selection_problems(catalog, selection)
        if problems:
            raise HTTPException(409, {"message": "selection is not runnable",
                                      "problems": [p.model_dump() for p in problems]})
        run = store.create_run({**selection, "next_run": str(path)},
                               body.note or f"next run from {rid} ({nr['model']}, {len(nr['cells'])} cells)")
        run_dir = var_dir / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.yaml").write_text(yaml.safe_dump(run.model_dump(), sort_keys=False))
        return run.model_dump()

    return r
```

- [ ] **Step 4: Pipeline changes** in `orchestrator/pipeline.py`:
  - Add `from dataengine.variation import load_cells`.
  - Add the constructor parameter `resolve_policy: Callable[[str], str] | None = None` and store it as `self._resolve_policy`.
  - Add these methods:

```python
    def run_defaults(self, run: Run, stage: StageDef) -> dict:
        """Param defaults that come from the run itself: a next run's generate targets its failing cells."""
        nr = run.selection.get("next_run")
        if stage.id != "generate" or not nr:
            return {}
        return {"split": f"cells:{nr}", "start": 0, "n": sum(int(c.get("k", 8)) for c in load_cells(nr))}

    def effective_params(self, run: Run, stage: StageDef, params: dict) -> dict:
        """User params over run defaults; `registry:<name>@<version|production>` resolved to an engine policy."""
        out = {**self.run_defaults(run, stage), **{k: v for k, v in params.items() if v is not None and v != ""}}
        policy = out.get("policy")
        if isinstance(policy, str) and policy.startswith("registry:"):
            ref = policy[len("registry:"):]
            if self._resolve_policy is None:
                raise PipelineError("no model registry configured", 400)
            try:
                out["policy"] = self._resolve_policy(ref)
            except (KeyError, ValueError) as e:
                raise PipelineError(f"cannot resolve {policy}: {e}", 400) from e
            if stage.id == "eval" and not out.get("model-version"):
                out["model-version"] = ref
        return out
```

  - In `stages()`, replace `"params": stage.param_dicts()` with:

```python
            defaults = self.run_defaults(run, stage)
            params = [{**p, "default": defaults.get(p["name"], p["default"])} for p in stage.param_dicts()]
```

  and use `"params": params`.
  - In `start()`, replace `flags = stage_argv(stage, params)` (inside the existing try) with `flags = stage_argv(stage, self.effective_params(run, stage, params))`. Keep storing the user's original `params` in the job.

- [ ] **Step 5: Control wiring** in `orchestrator/control.py`:
  - Add `eval_roots: list | None = None` to `create_control_app`.
  - Imports: `from serve.registry import Registry` and `from orchestrator.loop import create_loop_router`.
  - Create `registry = Registry(var_dir / "registry", repo_root=repo_root)` before the Pipeline, and pass `resolve_policy=registry.policy_for` to it. Set `app.state.registry = registry`.
  - Before the frontend mount, add:

```python
    roots = [Path(p) for p in eval_roots] if eval_roots else [repo_root / "reports", var_dir / "runs"]
    app.include_router(create_loop_router(store, registry, var_dir, repo_root, catalog_path, roots), prefix="/api")
```

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `uv run pytest -q tests/orchestrator` → all pass. Then `uv run pytest -q`.

- [ ] **Step 7: Real smoke** against the committed eval reports:
  1. Start `make dev`.
  2. `curl -s localhost:8700/api/evals | python -m json.tool | head -30`: 8 reports.
  3. Register SmolVLA:

```bash
curl -s -X POST localhost:8700/api/registry/register -H 'Content-Type: application/json' -d '{"ckpt_uri": "Rakeshdharan56/smolvla_panda_pick_v1", "dataset_versions": ["Rakeshdharan56/pick_robosuite@v1"], "eval_reports": {"robosuite": "reports/robosuite/eval_reports/robosuite/smolvla_panda_pick_v1/eval_report.json", "maniskill": "reports/robosuite/eval_reports/maniskill/smolvla_panda_pick_v1/eval_report.json"}}'
```

  4. Promote v1. Create a report from the same two reports, then create its next run. Check that `GET /api/runs/<id>/stages` shows generate with `split=cells:…`.
  5. Stop the server (kill by PID). Remove `var/registry` and `var/reports` from the smoke.

- [ ] **Step 8: Commit**

```bash
git add orchestrator tests/orchestrator/test_loop.py
git commit -m "feat(orchestrator): loop API (evals, registry, failure reports, next run) and registry: policies in the pipeline"
```

---
