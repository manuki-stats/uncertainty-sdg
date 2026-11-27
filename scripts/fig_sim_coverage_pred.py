"""Figure for the simulation chapter: coverage predicted from bias and the
variance ratio against observed coverage, one panel per generator.

Prediction (eq. coverage, Ch. 3): Phi(z*sqrt(rho) - lam) - Phi(-z*sqrt(rho) - lam),
with rho the variance ratio and lam = bias / empirical SD of the pooled
estimate, the SD recovered as sqrt(rmse^2 - bias^2). Step 1 uses the naive
interval, Steps 2-4 use T_alt. Reads only the committed summaries.

    python scripts/fig_sim_coverage_pred.py --summary results/summary --out figures/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import logging
import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import norm

CELLS = [
    ("synthpop plug-in", "synthpop_plugin__f6031a194e"),
    ("synthpop proper", "synthpop_proper__2ad55ad86b"),
    ("TVAE", "tvae__7c8fff32f1"),
    ("CTGAN ($R = 100$)", "ctgan__dfc3687fa3"),
    ("TabPFN", "tabpfn__01685b28f1"),
]
ESTIMANDS = {"E1": "#257de0", "E2": "#f47230", "E3": "#00b894"}   # validated, all pairs
STEPS = {1: "o", 2: "s", 3: "^", 4: "D"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


def predicted(row: pd.Series) -> float:
    var = max(row.rmse**2 - row.bias**2, 1e-15)
    lam = row.bias / np.sqrt(var)
    s = 1.96 * np.sqrt(row.var_ratio)
    return 100 * (norm.cdf(s - lam) - norm.cdf(-s - lam))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 4.4), sharex=True, sharey=True)
    for ax, (title, cell) in zip(axes.flat, CELLS):
        d = pd.read_csv(Path(a.summary) / f"{cell}.csv")
        d = d[((d.step == 1) & (d.estimator == "naive")) | ((d.step > 1) & (d.estimator == "T_alt"))]
        ax.plot([0, 100], [0, 100], color=TEXT2, lw=0.8, ls=(0, (3, 2)), zorder=1)
        ax.axhline(95, color=GRID, lw=0.8, zorder=0)
        ax.axvline(95, color=GRID, lw=0.8, zorder=0)
        for _, row in d.iterrows():
            ax.scatter(predicted(row), 100 * row.coverage, s=26, marker=STEPS[int(row.step)],
                       facecolor=ESTIMANDS[row.estimand], edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=8.5, color=TEXT, loc="left")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 95])
        ax.set_yticks([0, 25, 50, 75, 95])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("Predicted coverage (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed coverage (%)")

    leg = axes.flat[5]
    leg.axis("off")
    est = [Line2D([], [], ls="", marker="o", ms=6, mfc=c, mec="white", label=e) for e, c in ESTIMANDS.items()]
    stp = [Line2D([], [], ls="", marker=m, ms=6, mfc=TEXT2, mec="white",
                  label="Step 1 (naive)" if s == 1 else f"Step {s} ($T_{{\\mathrm{{alt}}}}$)")
           for s, m in STEPS.items()]
    l1 = leg.legend(handles=est, title="Estimand", loc="upper left", frameon=False, fontsize=8, title_fontsize=8)
    leg.add_artist(l1)
    leg.legend(handles=stp, title="Release design", loc="lower left", frameon=False, fontsize=8, title_fontsize=8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"sim_coverage_pred.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
