"""The propensity-score utility measure used for tuning (Woo et al. 2009;
Snoke et al. 2018).

Stack the confidential and the synthetic file, fit a logistic model for "row
came from the confidential file", and measure how far the fitted propensities
stray from the constant c = n_org / N:

    pMSE = mean_i (p_i - c)^2.

Which regressors the propensity model sees decides which failures pMSE can
detect. Near the null, the logistic score is (n_org n_syn / N) times the
difference in regressor means between the two files, so the model detects
mismatches in the means of its regressors and nothing else. We therefore use
as regressors the distinct monomials of the analysis model's sufficient
statistics X'X, X'y and y'y. A synthetic file that reproduces those moments
reproduces the point estimates and the residual variance of the analysis;
pMSE then measures the bias channel only. For the simulation (y ~ x1 * x2)
this gives ten regressors:

    x1, x2, x1:x2, x1^2, x1^2 x2, y, y x1, y x2, y x1 x2, y^2.

Null reference (Snoke et al. 2018), with k parameters including the intercept:

    E0 = (k - 1)(1 - c)^2 c / N,   SD0 = sqrt(2(k - 1)) (1 - c)^2 c / N.

Two properties matter for how we use it:
  - At fixed N, c and k, the raw pMSE, the ratio pMSE / E0 and the standardised
    (pMSE - E0) / SD0 rank configurations identically. We report the ratio.
  - The null value depends on the release design. On the simulation the
    plug-in oracle scores about 1.06 and the proper oracle about 2.04 (n = 5000,
    100 replications), because parameter noise is itself a moment mismatch. pMSE ranks configurations WITHIN a generator; it must never rank
    fidelity across generators with different synthesis designs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import minimize


class MomentFeatures:
    """Propensity-model regressors: the distinct monomials of the analysis
    model's sufficient statistics.

    Which monomials are distinct is decided once, on the confidential data, so
    k is fixed by the design and identical for every synthetic file scored
    against it (a binary x2 makes x2^2 = x2, for example, and that duplicate is
    dropped once, not re-decided per file).
    """

    def __init__(self, formula: str, tol: float = 1e-10):
        lhs, rhs = formula.split("~")
        self.response = lhs.strip()
        self.rhs = rhs.strip()
        self.tol = tol

    def fit(self, data: pd.DataFrame) -> "MomentFeatures":
        X = patsy.dmatrix(self.rhs, data, return_type="dataframe")
        self._design_info = X.design_info
        cols, names = self._candidates(X, data[self.response].to_numpy(dtype=float))
        keep: list[int] = []
        for j in range(cols.shape[1]):
            v = cols[:, j]
            if np.std(v) <= self.tol:                       # constant: absorbed by the intercept
                continue
            if any(np.allclose(v, cols[:, i], rtol=0, atol=self.tol * (1 + np.abs(v).max()))
                   for i in keep):
                continue                                    # structural duplicate
            keep.append(j)
        self._keep = keep
        self.names = [names[j] for j in keep]
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        (X,) = patsy.build_design_matrices([self._design_info], data,
                                           return_type="dataframe")
        cols, _ = self._candidates(X, data[self.response].to_numpy(dtype=float))
        return cols[:, self._keep]

    @property
    def k(self) -> int:
        """Parameters in the propensity model, intercept included."""
        return len(self._keep) + 1

    def _candidates(self, X: pd.DataFrame, y: np.ndarray):
        Xv = X.to_numpy(dtype=float)
        xn = list(X.columns)
        cols, names = [], []
        for i in range(Xv.shape[1]):                        # X'X
            for j in range(i, Xv.shape[1]):
                cols.append(Xv[:, i] * Xv[:, j])
                names.append(_product_name(xn[i], xn[j]))
        for i in range(Xv.shape[1]):                        # X'y
            cols.append(Xv[:, i] * y)
            names.append(_product_name(xn[i], self.response))
        cols.append(y * y)                                  # y'y
        names.append(_product_name(self.response, self.response))
        return np.column_stack(cols), names


def _product_name(a: str, b: str) -> str:
    if a == "Intercept":
        return b
    if b == "Intercept":
        return a
    return f"{a}*{b}"


@dataclass(frozen=True)
class PmseResult:
    pmse: float
    ratio: float          # pMSE / E0; about 1 for a correct plug-in fit, 2 for proper
    standardised: float   # (pMSE - E0) / SD0
    k: int
    n_org: int
    n_syn: int


def _logistic_probs(F: np.ndarray, s: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Maximum-likelihood logistic fit with a negligible ridge, which keeps the
    optimiser finite if a synthetic file is so bad the two files separate."""
    Z = np.column_stack([np.ones(len(F)), F])

    def loss(beta):
        eta = Z @ beta
        p = 0.5 * (1.0 + np.tanh(0.5 * eta))                # stable sigmoid
        nll = np.sum(np.logaddexp(0.0, eta) - s * eta) + 0.5 * ridge * beta[1:] @ beta[1:]
        grad = Z.T @ (p - s)
        grad[1:] += ridge * beta[1:]
        return nll, grad

    c = s.mean()
    beta0 = np.zeros(Z.shape[1])
    beta0[0] = np.log(c / (1 - c))
    res = minimize(loss, beta0, jac=True, method="L-BFGS-B",
                   options={"maxiter": 1000, "gtol": 1e-8})
    eta = Z @ res.x
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


def pmse_score(orig: pd.DataFrame, syn: pd.DataFrame, features: MomentFeatures) -> PmseResult:
    """pMSE of one synthetic file against the confidential file it was fitted to.
    `features` must already be fitted on `orig`."""
    Fo, Fs = features.transform(orig), features.transform(syn)
    n_org, n_syn = len(Fo), len(Fs)
    k = features.k
    if not np.all(np.isfinite(Fs)):
        return PmseResult(np.inf, np.inf, np.inf, k, n_org, n_syn)

    F = np.vstack([Fo, Fs])
    F = (F - F.mean(axis=0)) / F.std(axis=0)                 # conditioning only; fit is invariant
    s = np.r_[np.ones(n_org), np.zeros(n_syn)]
    N = n_org + n_syn
    c = n_org / N
    p = _logistic_probs(F, s)
    pmse = float(np.mean((p - c) ** 2))
    e0 = (k - 1) * (1 - c) ** 2 * c / N
    sd0 = np.sqrt(2 * (k - 1)) * (1 - c) ** 2 * c / N
    return PmseResult(pmse, pmse / e0, (pmse - e0) / sd0, k, n_org, n_syn)
