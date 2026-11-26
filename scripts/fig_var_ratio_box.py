"""Appendix figure: per-run variance ratios T / Var(q_bar), one box per step,
one panel per generator and estimand (after Drechsler 2018, Figs. 1-3).

Each run's ratio divides its variance estimate by the empirical variance of
the pooled estimate across replications of the same cell, step, estimand and
estimator, exactly as src/evaluation/summarise.py does; the white diamond is
the mean of the per-run ratios, i.e. the var_ratio column of the committed
summaries (the variance ratio reported in the tables). Step 1 uses the naive
interval, Steps 2-4 use T_alt. Reads the raw chunks in results/raw, which are
git-ignored, so run it where the raw results live:

    python scripts/fig_var_ratio_box.py --raw results/raw --out figures/

Check: the script prints max |mean ratio - var_ratio| against the summaries
when --summary is given; it should be ~1e-12.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

CELLS = [
    ("synthpop plug-in", "synthpop_plugin__f6031a194e"),
    ("synthpop proper", "synthpop_proper__2ad55ad86b"),
    ("TVAE", "tvae__7c8fff32f1"),
    ("CTGAN ($R = 100$)", "ctgan__dfc3687fa3"),
    ("TabPFN", "tabpfn__01685b28f1"),
]
ESTIMANDS = {"E1": "#257de0", "E2": "#f47230", "E3": "#00b894"}   # as in fig_sim_coverage_pred
LABELS = {"E1": "E1: mean", "E2": "E2: main slope", "E3": "E3: interaction"}
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"


def read_cell(raw: Path, cell: str) -> pd.DataFrame:
    files = sorted((raw / cell).glob("reps_*.csv"))
    if not files:
        raise FileNotFoundError(f"no chunks in {raw / cell}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def per_run_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of (step, estimand, ratio): naive in Step 1, T_alt in Steps 2-4."""
    keep = ((df.step == 1) & (df.estimator == "naive")) | \
           ((df.step > 1) & (df.estimator == "T_alt"))
    d = df[keep].dropna(subset=["variance"]).copy()
    err = d.estimate - d.target_value
    emp = err.groupby([d.step, d.estimand]).transform(lambda e: e.var(ddof=1))
    d["ratio"] = d.variance / emp
    return d[["step", "estimand", "ratio"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--summary", default=None, help="optional: check means against summaries")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {cell: per_run_ratios(read_cell(raw, cell)) for _, cell in CELLS}

    if a.summary:
        worst = 0.0
        for cell, d in data.items():
            s = pd.read_csv(Path(a.summary) / f"{cell}.csv")
            s = s[((s.step == 1) & (s.estimator == "naive")) |
                  ((s.step > 1) & (s.estimator == "T_alt"))]
            m = d.groupby(["step", "estimand"]).ratio.mean().rename("mine")
            j = s.set_index(["step", "estimand"]).join(m)
            worst = max(worst, float((j.mine - j.var_ratio).abs().max()))
        print(f"max |mean per-run ratio - summary var_ratio| = {worst:.2e}")

    allr = pd.concat(data.values()).ratio
    allr = allr[allr > 0]
    lo = 10 ** np.floor(np.log10(allr.quantile(0.001)))
    hi = 10 ** np.ceil(np.log10(allr.quantile(0.999)))

    plt.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(3, len(CELLS), figsize=(6.3, 5.6), sharex=True, sharey=True)
    n_out = 0
    for i, (est, col) in enumerate(ESTIMANDS.items()):
        for j, (name, cell) in enumerate(CELLS):
            ax = axes[i, j]
            d = data[cell]
            series = [d[(d.step == s) & (d.estimand == est)].ratio.to_numpy() for s in (1, 2, 3, 4)]
            ax.axhline(1, color=TEXT2, lw=0.8, zorder=1)
            bp = ax.boxplot(series, positions=[1, 2, 3, 4], widths=0.62, patch_artist=True,
                            whis=1.5, showfliers=True, zorder=2,
                            medianprops=dict(color=TEXT, lw=1.0),
                            whiskerprops=dict(color=col, lw=0.8),
                            capprops=dict(color=col, lw=0.8),
                            boxprops=dict(edgecolor=col, lw=0.8),
                            flierprops=dict(marker="o", ms=1.6, mfc=col, mec="none", alpha=0.35))
            for b in bp["boxes"]:
                b.set_facecolor(matplotlib.colors.to_rgba(col, 0.18))
            means = [x.mean() for x in series]
            ax.scatter([1, 2, 3, 4], means, marker="D", s=14, facecolor="white",
                       edgecolor=TEXT, lw=0.7, zorder=4)
            for k, x in enumerate(series):          # points beyond the axis: count them
                above = int((x > hi).sum())
                if above:
                    n_out += above
                    ax.annotate(f"{above}↑", (k + 1, hi), xytext=(0, -2),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=6, color=TEXT2)
            ax.set_yscale("log")
            ax.set_ylim(lo, hi)
            ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            if i == 0:
                ax.set_title(name, fontsize=8, color=TEXT)
            if j == 0:
                ax.set_ylabel(LABELS[est], color=TEXT)
            ax.set_xticks([1, 2, 3, 4])
            if i == 2:
                ax.set_xlabel("Step", color=TEXT2)
    for ax in axes.flat:                    # after all set_yscale calls (shared axis)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    fig.supylabel(r"$T\,/\,\widehat{\mathrm{Var}}(\bar q_m)$ per run (log scale)",
                  fontsize=8, color=TEXT, x=0.005)
    fig.tight_layout(rect=(0.02, 0, 1, 1), h_pad=0.6, w_pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"sim_var_ratio_box.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}/sim_var_ratio_box.pdf/.png; {n_out} runs outside the axis (counted in panels)")


if __name__ == "__main__":
    main()
