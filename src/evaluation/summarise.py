"""Summarise one cell: coverage with Wilson interval, width, bias, variance ratio,
kappa, correction rate, dispersion. Reads the raw chunks, writes one CSV.

Usage:
  python -m src.evaluation.summarise --config configs/oracle_proper.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.evaluation.metrics import needs_escalation, rmse_foil, wilson_interval
from src.results_io import read_cell

SUMMARY_ROOT = Path("results/summary")


def summarise(df: pd.DataFrame, r: float) -> pd.DataFrame:
    out = []
    for (step, est, estimator), g in df.groupby(["step", "estimand", "estimator"], sort=True):
        g = g.dropna(subset=["variance"])
        n = len(g)
        hits = int(g["covered"].sum())
        lo, hi = wilson_interval(hits, n)
        err = g["estimate"] - g["target_value"]
        emp_var = float(err.var(ddof=1))           # empirical Var(q_bar - target)
        row = {
            "step": step, "estimand": est, "estimator": estimator, "R": n,
            "coverage": hits / n, "wilson_lo": lo, "wilson_hi": hi,
            "escalate": needs_escalation(hits, n),
            "mean_width": g["width"].mean(),
            "bias": err.mean(),
            "rmse": rmse_foil(g["estimate"], g["target_value"]),
            "var_ratio": g["variance"].mean() / emp_var,
            "var_ratio_median": (g["variance"] / emp_var).median(),
            "kappa": g["kappa"].mean(),
            "kappa_norm": (g["kappa"].mean() - 1) / r,
            "corrected_rate": g["corrected"].mean(),
        }
        for c in [c for c in g.columns if c.startswith("disp_")]:
            row[c] = g[c].mean()
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--estimator", default=None, help="print only this estimator")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    s = summarise(read_cell(cfg.name, cfg.hash()), cfg.r)

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    s.to_csv(SUMMARY_ROOT / f"{cfg.name}__{cfg.hash()}.csv", index=False)

    view = s if args.estimator is None else s[s.estimator.isin([args.estimator, "naive"])]
    cols = ["step", "estimand", "estimator", "R", "coverage", "wilson_lo", "wilson_hi",
            "var_ratio", "bias", "kappa", "corrected_rate"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
