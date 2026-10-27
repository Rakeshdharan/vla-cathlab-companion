"""README figures from the committed cross-physics results (reports/robosuite/cross_physics_v1.json).

    uv run --with matplotlib python -m scripts.plot_results [--out docs/assets/plots]

Writes cross_physics_heatmap.png (success per model x eval engine), loop_retrain.png (X-VLA v2 -> loop v3, Wilson 95 %
CI) and failure_stages.png (where the production model fails, per engine).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/robosuite/cross_physics_v1.json"
ENGINES = {"robosuite": "robosuite\n(MuJoCo)", "maniskill": "ManiSkill3\n(PhysX 5)", "genesis": "Genesis"}
MODELS = {  # row order; label, trained on
    "expert": ("Scripted expert", "—"),
    "act_panda_pick_v1": ("ACT", "robosuite"),
    "smolvla_panda_pick_v1": ("SmolVLA  (production v2)", "robosuite"),
    "xvla_panda_pick_v1": ("X-VLA", "robosuite"),
    "smolvla_panda_maniskill_v1": ("SmolVLA", "maniskill"),
    "xvla_panda_maniskill_v1": ("X-VLA", "maniskill"),
    "xvla_panda_genesis_v1": ("X-VLA", "genesis"),
    "xvla_panda_pick_v2": ("X-VLA v2", "all three"),
    "xvla_panda_pick_v3": ("X-VLA loop v3", "all three"),
}
PRODUCTION = "smolvla_panda_pick_v1"
STAGES = ("reach", "grasp", "lift", "transport", "place")

# Reference data-viz palette (validated light-mode categorical slots, blue sequential ramp, ink tokens).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
BLUES = ["#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _style(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    })


def heatmap(plt, data: dict, out: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    rows = list(MODELS)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    grid = [[data[f"{e}/{m}"]["overall"]["success_rate"] for e in ENGINES] for m in rows]
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, m in enumerate(rows):
        trained = MODELS[m][1]
        for j, e in enumerate(ENGINES):
            o = data[f"{e}/{m}"]["overall"]
            in_domain = e in trained or trained == "all three"
            ax.text(j, i, f"{o['success_rate']:.0%}\n{o['ci_low']:.0%}–{o['ci_high']:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if in_domain else "normal",
                    color="#ffffff" if o["success_rate"] > 0.55 else INK)
    ax.set_xticks(range(len(ENGINES)), list(ENGINES.values()))
    ax.set_yticks(range(len(rows)), [f"{MODELS[m][0]}  ·  {MODELS[m][1]}" for m in rows])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(ENGINES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.set_title("Success rate by eval engine (rows: model · trained on)", pad=36)
    fig.text(0.01, 0.01, "suite_v2: 3 objects × 5 tiers × 4 seeds, n = 60 per cell, Wilson 95 % CI. Bold = in-domain.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "cross_physics_heatmap.png", dpi=150)
    plt.close(fig)


def loop_retrain(plt, data: dict, out: Path) -> None:
    series = [("xvla_panda_pick_v2", "X-VLA v2 (before loop)"), ("xvla_panda_pick_v3", "X-VLA v3 (after one loop)")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width, gap = 0.36, 0.02
    for k, (m, label) in enumerate(series):
        xs = [j + (k - 0.5) * (width + gap) for j in range(len(ENGINES))]
        os_ = [data[f"{e}/{m}"]["overall"] for e in ENGINES]
        rates = [o["success_rate"] for o in os_]
        err = [[r - o["ci_low"] for r, o in zip(rates, os_, strict=True)],
               [o["ci_high"] - r for r, o in zip(rates, os_, strict=True)]]
        ax.bar(xs, rates, width, color=SERIES[k], label=label, zorder=2)
        ax.errorbar(xs, rates, yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
        for x, r, o in zip(xs, rates, os_, strict=True):
            ax.text(x, o["ci_high"] + 0.015, f"{r:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    prod = data[f"robosuite/{PRODUCTION}"]["overall"]["success_rate"]
    ax.axhline(prod, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(ENGINES) - 0.5, prod + 0.012, f"production v2 on robosuite: {prod:.0%}", ha="right", fontsize=8.5,
            color=INK_2)
    ax.set_xticks(range(len(ENGINES)), [v.replace("\n", " ") for v in ENGINES.values()])
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.set_title("One loop iteration (+48 targeted episodes)")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "loop_retrain.png", dpi=150)
    plt.close(fig)


def failure_stages(plt, data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    engines = list(ENGINES)
    for i, e in enumerate(engines):
        d = data[f"{e}/{PRODUCTION}"]
        n = d["overall"]["n"]
        left = 0.0
        succ = round(d["overall"]["success_rate"] * n)
        segments = [("success", succ, "#c3c2b7")] + [(s, d["failed_stage"][s], SERIES[k]) for k, s in enumerate(STAGES)]
        for name, count, color in segments:
            if not count:
                continue
            w = count / n
            ax.barh(i, w - 0.004, left=left, height=0.56, color=color, zorder=2)
            if w >= 0.07:
                ax.text(left + w / 2, i, f"{name}\n{count}", ha="center", va="center", fontsize=8.5,
                        color=INK if color in ("#c3c2b7", "#eda100", "#e87ba4", "#1baf7a") else "#ffffff")
            left += w
    ax.set_yticks(range(len(engines)), [v.replace("\n", " ") for v in ENGINES.values()])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Where the production model fails (SmolVLA, trained on robosuite)", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c3c2b7")] + [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
    ax.legend(handles, ["success", *[f"fails at {s}" for s in STAGES]], frameon=False, ncol=3, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.45, -0.12))
    fig.tight_layout()
    fig.savefig(out / "failure_stages.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--out", default=str(ROOT / "docs/assets/plots"))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    data = json.loads(Path(a.results).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    heatmap(plt, data, out)
    loop_retrain(plt, data, out)
    failure_stages(plt, data, out)
    print(f"wrote {', '.join(sorted(p.name for p in out.glob('*.png')))} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
