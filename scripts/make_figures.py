"""Draw the result figures used in the README and the Phase 1 report.

    python scripts/make_figures.py      # reads results/*.csv, writes results/figures/*.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})

SHOWN = ["greedy-weighted+orch", "greedy-weighted", "greedy-latency", "batch-milp", "drl-ddqn", "all-cloud"]


def grouped_bars(summary, metric, xlabel, fname, fmt):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    y = np.arange(len(SHOWN))
    h = 0.36
    for k, (cond, color) in enumerate((("normal", BLUE), ("wan-outage", ORANGE))):
        vals = [summary.loc[(cond, s), metric] for s in SHOWN]
        bars = ax.barh(y + (k - 0.5) * h, vals, height=h - 0.04, color=color, label=cond, edgecolor=SURFACE,
                       linewidth=1)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt(v), va="center", fontsize=7.5,
                    color=INK2)
    ax.set_yticks(y, SHOWN)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / fname, dpi=200)
    plt.close(fig)


def forecast_fig():
    df = pd.read_csv(ROOT / "results" / "forecast_eval.csv")
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for model, color in zip(df.model.unique(), (INK2, BLUE, ORANGE, AQUA)):
        d = df[df.model == model]
        ax.plot(d.h, d.mae, marker="o", markersize=5, linewidth=2, color=color, label=model)
    ax.set_xticks([1, 5, 10])
    ax.set_xlabel("forecast horizon (minutes ahead)")
    ax.set_ylabel("MAE (arrivals / min)")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_mae.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(ROOT / "results" / "baseline_summary.csv").set_index(["condition", "scheduler"])
    grouped_bars(summary, "mean_weighted_cost", "mean weighted cost J per task (lower is better)",
                 "baseline_weighted_cost.png", lambda v: f"{v:.2f}")
    grouped_bars(summary, "deadline_miss_rate", "deadline miss rate (lower is better)",
                 "baseline_miss_rate.png", lambda v: f"{100 * v:.1f}%")
    forecast_fig()
    print("figures in", FIG)


if __name__ == "__main__":
    sys.exit(main())
