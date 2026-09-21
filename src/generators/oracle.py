"""The parametric oracle: a correctly specified generator with known behaviour.

Not a method under test. It validates the harness and the combining rules
before any neural network is involved, and it isolates the release designs in
their pure form, with no model error.

The model is sequential, like synthpop:
    x2 ~ Bernoulli(p),  x1 | x2 ~ N(mu_{x2}, tau^2),  y | x1, x2 ~ N(X beta, sigma^2).
The DGP is the special case mu_0 = mu_1, so the oracle is correctly specified.
Letting mu depend on x2 also makes it SATURATED for E1-E3: the synthesiser's
implied estimates equal the complete-data estimates exactly, so the oracle is
congenial with the analysis and no efficiency difference contaminates the gate.
Each factor has a conjugate posterior under a reference prior, so draws are exact.

Three modes, differing only in where theta comes from:

  proper    fresh theta* from the posterior for EVERY file (Rubin's first stage)
  plugin    the MLE, no posterior draw at all (Reiter & Kinney 2012)
  ordinary  ONE posterior draw at fit time, frozen for every file

Under the SDA, plugin and proper are both valid with the right estimator;
ordinary is not, because a single draw displaces the estimate by a posterior
standard deviation that nothing in the release reflects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Generator

MODES = ("proper", "plugin", "ordinary")


class ParametricOracle(Generator):
    name = "oracle"

    def __init__(self, mode: str = "proper"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode

    def fit(self, data: pd.DataFrame, *, train_seed: int) -> "ParametricOracle":
        x1 = data["x1"].to_numpy(dtype=float)
        x2 = data["x2"].to_numpy(dtype=float)
        y = data["y"].to_numpy(dtype=float)
        n = y.size
        self._n = n

        successes = float(x2.sum())
        self._a, self._b = 0.5 + successes, 0.5 + n - successes   # Jeffreys
        g1 = x2 == 1
        self._ng = np.array([n - g1.sum(), g1.sum()], dtype=float)
        self._xbar = np.array([x1[~g1].mean(), x1[g1].mean()])
        ss = ((x1[~g1] - self._xbar[0]) ** 2).sum() + ((x1[g1] - self._xbar[1]) ** 2).sum()
        self._s2 = float(ss / (n - 2))                               # pooled within-group

        X = self._design(x1, x2)
        xtx_inv = np.linalg.inv(X.T @ X)
        self._beta_hat = xtx_inv @ X.T @ y
        resid = y - X @ self._beta_hat
        self._rss = float(resid @ resid)
        self._k = X.shape[1]
        self._chol = np.linalg.cholesky(xtx_inv)

        if self.mode == "ordinary":
            self._frozen = self._draw_theta(np.random.default_rng(train_seed))
        return self

    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(gen_seed)
        if self.mode == "proper":
            p, mu, tau2, beta, sigma2 = self._draw_theta(rng)
        elif self.mode == "plugin":
            p, mu, tau2, beta, sigma2 = self._mle()
        else:
            p, mu, tau2, beta, sigma2 = self._frozen

        x2 = rng.binomial(1, p, size=n).astype(float)
        x1 = mu[x2.astype(int)] + np.sqrt(tau2) * rng.normal(size=n)
        y = (beta[0] + beta[1] * x1 + beta[2] * x2 + beta[3] * x1 * x2
             + rng.normal(scale=np.sqrt(sigma2), size=n))
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def _draw_theta(self, rng: np.random.Generator):
        p = rng.beta(self._a, self._b)
        tau2 = (self._n - 2) * self._s2 / rng.chisquare(self._n - 2)
        mu = self._xbar + np.sqrt(tau2 / self._ng) * rng.normal(size=2)
        sigma2 = self._rss / rng.chisquare(self._n - self._k)
        beta = self._beta_hat + np.sqrt(sigma2) * (self._chol @ rng.normal(size=self._k))
        return p, mu, tau2, beta, sigma2

    def _mle(self):
        p_hat = (self._a - 0.5) / self._n
        return (p_hat, self._xbar, self._s2, self._beta_hat,
                self._rss / (self._n - self._k))

    @staticmethod
    def _design(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(x1), x1, x2, x1 * x2])
