"""The estimand battery E1-E3, computed on each synthetic dataset (cheap).

    E1  univariate mean of a focal variable   --- fit litmus (determinism can't bias a mean)
    E2  coefficient of a main predictor        --- undercoverage
    E3  hard estimand (interaction / low-density) --- bias + undercoverage

Each returns a point estimate and its within-dataset variance (q, u), the two
inputs the combining rules need.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

Estimate = tuple[float, float]  # (q, u)

_FAILED: Estimate = (float("nan"), float("nan"))


def e1_mean(df: pd.DataFrame, var: str) -> Estimate:
    """Point estimate and variance of a univariate mean."""
    y = pd.to_numeric(df[var], errors="coerce").to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 2:
        return _FAILED
    return float(y.mean()), float(y.var(ddof=1) / y.size)


def e2_coefficient(df: pd.DataFrame, formula: str, term: str) -> Estimate:
    """A main-predictor regression coefficient and its variance."""
    return _ols_term(df, formula, term)


def e3_hard(df: pd.DataFrame, formula: str, term: str) -> Estimate:
    """An interaction / low-density coefficient and its variance."""
    return _ols_term(df, formula, term)


def _ols_term(df: pd.DataFrame, formula: str, term: str) -> Estimate:
    """Fit the analysis model and return (coefficient, squared standard error).

    A degenerate fit returns NaN rather than raising: a generator that collapses
    a column to a constant is a result to be logged, not a crash. A missing term
    name, by contrast, is a specification error on our side and raises.
    """
    try:
        fit = smf.ols(formula, data=df).fit()
    except Exception:
        return _FAILED

    if term not in fit.params.index:
        raise KeyError(
            f"term {term!r} not in the fitted model; available: {list(fit.params.index)}"
        )

    q, se = float(fit.params[term]), float(fit.bse[term])
    if not (np.isfinite(q) and np.isfinite(se)):
        return _FAILED
    return q, se ** 2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + 1.5 * x1 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    checks = [
        ("E1", e1_mean(df, "y")),
        ("E2", e2_coefficient(df, "y ~ x1 * x2", "x1")),
        ("E3", e3_hard(df, "y ~ x1 * x2", "x1:x2")),
    ]
    for name, (q, u) in checks:
        print(f"{name}: q = {q: .4f}   u = {u:.6f}   se = {np.sqrt(u):.4f}")

def estimands_fast(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """E1-E3 for the simulation design, without the formula interface.

    Hard-coded to the design matrix [1, x1, x2, x1*x2] of the simulation DGP,
    which is what makes it ~17x faster than the patsy path: no formula is
    parsed. It is NOT general --- the ACS analysis will need either the
    functions above or its own fast path. test_estimands.py pins it against the
    statsmodels route, so the two cannot drift apart unnoticed.

    Returns (q, u) as length-3 arrays ordered E1, E2, E3.
    """
    y = df["y"].to_numpy(dtype=float)
    x1 = df["x1"].to_numpy(dtype=float)
    x2 = df["x2"].to_numpy(dtype=float)
    n = y.size

    X = np.column_stack([np.ones(n), x1, x2, x1 * x2])
    G = np.linalg.inv(X.T @ X)
    beta = G @ (X.T @ y)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - X.shape[1])
    var_beta = s2 * np.diag(G)

    q = np.array([y.mean(), beta[1], beta[3]])
    u = np.array([y.var(ddof=1) / n, var_beta[1], var_beta[3]])
    return q, u