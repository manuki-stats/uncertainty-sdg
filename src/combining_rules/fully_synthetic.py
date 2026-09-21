"""Fully-synthetic combining rules (Raghunathan-Reiter-Rubin 2003; Reiter 2002).

Given m synthetic datasets, each yielding a point estimate q_i and its
within-dataset variance u_i, these rules pool them into a single estimate,
a total variance, and a confidence interval --- the machinery whose coverage
the whole thesis measures.

    qbar = mean(q_i)
    bm   = between-dataset variance of the q_i
    ubar = mean(u_i)
    T    = (1 + 1/m) * bm  -  ubar          # fully-synthetic total variance
    nu   = (m - 1) * (1 - ubar / ((1 + 1/m) * bm))**2

The subtraction can drive T below zero, most often at small m. We then fall back
on the Reiter (2002) adjustment T = (n_syn / n_org) * ubar and use a normal
quantile, flagging the row so the frequency of the fallback can be reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PooledInference:
    estimate: float          # qbar
    total_variance: float    # T (post-correction)
    df: float                # degrees of freedom
    ci_low: float
    ci_high: float
    negative_variance: bool = False   # True if the correction fired
    b_m: float = float("nan")         # between-dataset variance, logged raw
    u_bar: float = float("nan")       # mean within-dataset variance

    def covers(self, truth: float) -> bool:
        """Whether the interval contains the known truth (simulation only)."""
        return self.ci_low <= truth <= self.ci_high

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def combine_fully_synthetic(
    q,
    u,
    m: int,
    alpha: float = 0.05,
    n_syn: int | None = None,
    n_org: int | None = None,
) -> PooledInference:
    """Pool m estimates under the fully-synthetic rules.

    Args:
        q: length-m point estimates, one per synthetic dataset.
        u: length-m within-dataset variances.
        m: number of synthetic datasets.
        alpha: 1 - nominal coverage (0.05 for 95% intervals).
        n_syn: rows per synthetic dataset, needed only by the correction.
        n_org: rows in the original data, needed only by the correction.

    Any non-finite q or u makes the whole cell unusable, so we return NaNs and
    let the harness log the failure rather than pool a corrupted estimate.
    """
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    if q.size != m or u.size != m:
        raise ValueError(f"expected {m} estimates, got q={q.size}, u={u.size}")
    if m < 2:
        raise ValueError("fully-synthetic pooling needs m >= 2 to form b_m")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(u))):
        nan = float("nan")
        return PooledInference(nan, nan, nan, nan, nan, False, nan, nan)

    q_bar = float(q.mean())
    b_m = float(q.var(ddof=1))
    u_bar = float(u.mean())
    T = (1.0 + 1.0 / m) * b_m - u_bar

    if T > 0:
        df = (m - 1) * (1.0 - u_bar / ((1.0 + 1.0 / m) * b_m)) ** 2
        crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        negative = False
    else:
        if n_syn is None or n_org is None:
            raise ValueError(
                "total variance is non-positive and the correction needs "
                "n_syn and n_org; pass them from the config"
            )
        T = (n_syn / n_org) * u_bar
        df = float("inf")
        crit = stats.norm.ppf(1.0 - alpha / 2.0)
        negative = True

    half = crit * np.sqrt(T)
    return PooledInference(
        estimate=q_bar,
        total_variance=float(T),
        df=float(df),
        ci_low=float(q_bar - half),
        ci_high=float(q_bar + half),
        negative_variance=negative,
        b_m=b_m,
        u_bar=u_bar,
    )