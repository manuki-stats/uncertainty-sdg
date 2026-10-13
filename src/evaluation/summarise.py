"""Summarise one cell: coverage with Wilson interval, width, bias, variance ratio,
kappa, correction rate, dispersion. Reads the raw chunks, writes one CSV.

Usage:
  python -m src.evaluation.summarise --config configs/oracle_proper.yaml
  python -m src.evaluation.summarise --config configs/acs_tvae.yaml --target sample

--target sample re-pools the stored per-file estimates against each
replication's complete-data estimate q_org (sampling term dropped; rules.py,
target "sample"), for a cell run with target "population". It needs the
orig_*.csv tables, so only chunks written from 13 Oct 2026 on qualify. The
result is identical to running the cell with target "sample" (tests/test_acs.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.combining_rules.rules import naive, pool
from src.evaluation.metrics import needs_escalation, rmse_foil, wilson_interval
from src.results_io import read_cell, read_cell_files, read_cell_orig

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


def sample_target_rows(cfg: ExperimentConfig) -> pd.DataFrame:
    """Result rows against q_org, rebuilt from a population-target cell's
    per-file (files_*) and confidential-sample (orig_*) tables."""
    if cfg.target != "population":
        raise ValueError("cell already has target 'sample'; summarise it directly")
    h = cfg.hash()
    files = read_cell_files(cfg.name, h)
    q_org = read_cell_orig(cfg.name, h).set_index(["rep", "estimand"])["q"]
    rows = []
    for (rep, step, est), g in files.groupby(["rep", "step", "estimand"], sort=True):
        if (rep, est) not in q_org.index:
            continue                        # chunk written before orig_* existed
        target = float(q_org[(rep, est)])
        g = g.sort_values("file")
        if step == 1:
            ivs = [naive(float(g["q"].iloc[0]), float(g["u"].iloc[0]), cfg.alpha)]
            extra = {"b": np.nan, "u_bar": float(g["u"].iloc[0]), "kappa": np.nan,
                     "files": 1}
        else:
            p = pool(g["q"].to_numpy(), g["u"].to_numpy(), r=cfg.r, alpha=cfg.alpha,
                     target="sample", estimators=cfg.estimators)
            ivs = list(p.intervals.values())
            extra = {"b": p.b, "u_bar": p.u_bar, "kappa": p.kappa, "files": p.m}
        for iv in ivs:
            rows.append({"rep": rep, "step": step, "estimand": est, **extra,
                         "estimator": iv.estimator, "target_value": target,
                         "estimate": iv.estimate, "variance": iv.variance, "df": iv.df,
                         "lo": iv.lo, "hi": iv.hi, "width": iv.width,
                         "covered": iv.covers(target), "corrected": iv.corrected})
    if not rows:
        raise FileNotFoundError(f"no replications of {cfg.name} have orig_* estimates")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--estimator", default=None, help="print only this estimator")
    ap.add_argument("--target", choices=["config", "sample"], default="config",
                    help="'sample': re-pool against q_org (population-target cells only)")
    args = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if args.target == "sample":
        s = summarise(sample_target_rows(cfg), cfg.r)
        suffix = "__qorg"
    else:
        s = summarise(read_cell(cfg.name, cfg.hash()), cfg.r)
        suffix = ""

    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    s.to_csv(SUMMARY_ROOT / f"{cfg.name}__{cfg.hash()}{suffix}.csv", index=False)

    view = s if args.estimator is None else s[s.estimator.isin([args.estimator, "naive"])]
    cols = ["step", "estimand", "estimator", "R", "coverage", "wilson_lo", "wilson_hi",
            "var_ratio", "bias", "kappa", "corrected_rate"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(view[cols].to_string(index=False))


if __name__ == "__main__":
    main()
