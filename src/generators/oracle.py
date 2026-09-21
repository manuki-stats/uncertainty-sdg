"""The parametric oracle: a correctly-specified generator with known behaviour.

This is NOT a method under test. It is a generator we know is correct, used to
validate the harness: under the fully-synthetic combining rules it must yield
~95% coverage on E1-E3. If it does, the combining-rules code is trustworthy and
every later result stands on solid ground. If it does not, the bug is ours, in
the variance code --- and we find it before any neural net is involved.

The model factorises as p(x2) p(x1) p(y | x1, x2), matching the simulation DGP
exactly, and each factor has a conjugate posterior under a reference prior, so
the parameter draws are exact and need no MCMC.

What makes this PROPER synthesis, and the whole point of the oracle: every call
to generate() draws a fresh theta* from the posterior before generating. Plug in
the MLE instead and the synthetic datasets carry sampling noise but no parameter
uncertainty, b_m collapses, and the oracle under-covers --- the very failure we
expect from a neural generator held at one frozen fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Generator


class ParametricOracle(Generator):
    name = "oracle"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def fit(self, data: pd.DataFrame) -> "ParametricOracle":
        """Compute the posterior of every parameter. No draws happen here."""
        n = len(data)
        x1 = data["x1"].to_numpy(dtype=float)
        x2 = data["x2"].to_numpy(dtype=float)
        y = data["y"].to_numpy(dtype=float)

        self.n = n

        # x2 ~ Bernoulli(p), Jeffreys prior Beta(1/2, 1/2)
        successes = float(x2.sum())
        self._a = 0.5 + successes
        self._b = 0.5 + n - successes

        # x1 ~ N(mu, tau^2), reference prior p(mu, tau^2) ~ 1/tau^2
        self._xbar = float(x1.mean())
        self._s2 = float(x1.var(ddof=1))

        # y | x1, x2 ~ N(X beta, sigma^2), reference prior p(beta, sigma^2) ~ 1/sigma^2
        X = self._design(x1, x2)
        xtx_inv = np.linalg.inv(X.T @ X)
        self._beta_hat = xtx_inv @ X.T @ y
        resid = y - X @ self._beta_hat
        self._rss = float(resid @ resid)
        self._k = X.shape[1]
        self._chol = np.linalg.cholesky(xtx_inv)
        return self

    def _draw_theta(self):
        """One draw from the joint posterior of every parameter."""
        rng = self.rng
        p = rng.beta(self._a, self._b)
        tau2 = (self.n - 1) * self._s2 / rng.chisquare(self.n - 1)
        mu = self._xbar + np.sqrt(tau2 / self.n) * rng.normal()
        sigma2 = self._rss / rng.chisquare(self.n - self._k)
        beta = self._beta_hat + np.sqrt(sigma2) * (self._chol @ rng.normal(size=self._k))
        return p, mu, tau2, beta, sigma2

    def _current_theta(self):
        """Proper synthesis draws fresh parameters for every dataset."""
        return self._draw_theta()

    def generate(self, n: int) -> pd.DataFrame:
        p, mu, tau2, beta, sigma2 = self._current_theta()
        rng = self.rng
        x1 = rng.normal(mu, np.sqrt(tau2), size=n)
        x2 = rng.binomial(1, p, size=n).astype(float)
        y = (
            beta[0] + beta[1] * x1 + beta[2] * x2 + beta[3] * x1 * x2
            + rng.normal(scale=np.sqrt(sigma2), size=n)
        )
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    @staticmethod
    def _design(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(x1), x1, x2, x1 * x2])

class OrdinaryOracle(ParametricOracle):
    """The oracle run as ORDINARY synthesis: theta* drawn once and then frozen.

    Same correctly-specified model, same posterior --- the only change is that
    all m datasets descend from a single parameter draw. That is the deep
    generators' situation at step 2, with model error removed, so whatever
    undercoverage appears here is attributable to nothing but the missing
    parameter uncertainty. It is the floor CTGAN's coverage should be read
    against, not a method under test.
    """
    name = "ordinary_oracle"

    def fit(self, data: pd.DataFrame) -> "OrdinaryOracle":
        super().fit(data)
        self._theta = self._draw_theta()
        return self

    def _current_theta(self):
        return self._theta