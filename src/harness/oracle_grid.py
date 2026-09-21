"""Appendix C diagnostic: how the combining rules behave across m and n_syn/n_org.

Runs the proper oracle (which must cover) and the ordinary oracle (which must
not) over a grid, recording coverage, width, degrees of freedom, and how often
the negative-variance correction fires. This is not the main study; it
characterises the apparatus before the generators arrive.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from src.combining_rules.fully_synthetic import combine_fully_synthetic
from src.data.simulation import SimulationDGP
from src.evaluation.estimands import estimands_fast
from src.evaluation.metrics import mc_standard_error
from src.generators.oracle import OrdinaryOracle, ParametricOracle

OUT = Path("results/oracle_grid.csv")
NAMES = ("E1", "E2", "E3")

M_VALUES = (10, 20, 30, 50)
R_RATIOS = (1, 2, 5, 10)
N_ORG = 10_000
N_ORG_CHECK = (1_000, 100_000)   # invariance check, at m=20 r=5 only
R_REPS = 500
ALPHA = 0.05


def run_cell(gen_cls, m, ratio, n_org, R, seed0=0):
    dgp = SimulationDGP()
    truth = np.array([dgp.truth()[k] for k in NAMES])
    n_syn = ratio * n_org

    cov = np.zeros(3); neg = np.zeros(3)
    width = np.zeros((R, 3)); dfs = np.full((R, 3), np.nan); est = np.zeros((R, 3))

    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        real = dgp.sample(n_org, rng)
        gen = gen_cls(rng).fit(real)

        Q = np.empty((m, 3)); U = np.empty((m, 3))
        for i in range(m):
            Q[i], U[i] = estimands_fast(gen.generate(n_syn))

        for j in range(3):
            res = combine_fully_synthetic(
                Q[:, j], U[:, j], m=m, alpha=ALPHA, n_syn=n_syn, n_org=n_org
            )
            cov[j] += res.covers(truth[j])
            neg[j] += res.negative_variance
            width[r, j] = res.width
            est[r, j] = res.estimate
            if np.isfinite(res.df):
                dfs[r, j] = res.df

    rows = []
    for j, name in enumerate(NAMES):
        c = cov[j] / R
        rows.append({
            "generator": gen_cls.name, "m": m, "ratio": ratio,
            "n_org": n_org, "n_syn": n_syn, "R": R, "estimand": name,
            "coverage": round(c, 4),
            "mc_se": round(mc_standard_error(c, R), 4),
            "median_width": round(float(np.median(width[:, j])), 5),
            "bias": round(float(est[:, j].mean() - truth[j]), 5),
            "neg_var_rate": round(neg[j] / R, 4),
            "median_df": round(float(np.nanmedian(dfs[:, j])), 2),
        })
    return rows


def main():
    cells = [(m, r, N_ORG) for m in M_VALUES for r in R_RATIOS]
    cells += [(20, 5, n) for n in N_ORG_CHECK]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows, t0 = [], time.time()

    for gen_cls in (ParametricOracle, OrdinaryOracle):
        for m, ratio, n_org in cells:
            t = time.time()
            all_rows += run_cell(gen_cls, m, ratio, n_org, R_REPS)
            print(f"{gen_cls.name:16s} m={m:<3d} r={ratio:<3d} n_org={n_org:<7d} "
                  f"{time.time() - t:6.1f}s", flush=True)

    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n{len(all_rows)} rows -> {OUT}   total {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()