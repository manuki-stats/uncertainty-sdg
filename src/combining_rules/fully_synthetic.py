"""Fully-synthetic combining rules (Raghunathan-Reiter-Rubin 2003; Reiter 2002).

Given m synthetic datasets, each yielding a point estimate q_i and its
within-dataset variance u_i, these rules pool them into a single estimate,
a total variance, and a confidence interval --- the machinery whose coverage
the whole thesis measures.

    qbar = mean(q_i)
    bm   = between-dataset variance of the q_i
    ubar = mean(u_i)
    T    = (1 + 1/m) * bm  -  ubar      # fully-synthetic total variance
    (the negative-variance correction applies when T <= 0)

Fill in the exact estimator and its degrees of freedom when we build the
harness; this stub fixes the interface and names the correction so the oracle
test has something to call.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class PooledInference:
    estimate: float          # qbar
    total_variance: float    # T (post-correction)
    df: float                # degrees of freedom
    ci_low: float
    ci_high: float

    def covers(self, truth: float) -> bool:
        """Whether the interval contains the known truth (simulation only)."""
        return self.ci_low <= truth <= self.ci_high

    @property
    def width(self) -> float:
        return self.ci_high - self.ci_low


def combine_fully_synthetic(q, u, m, alpha: float = 0.05) -> "PooledInference":
    """Pool m estimates under the fully-synthetic rules. TO IMPLEMENT.

    Args:
        q: length-m point estimates, one per synthetic dataset.
        u: length-m within-dataset variances.
        m: number of synthetic datasets.
        alpha: 1 - nominal coverage (0.05 for 95% intervals).
    """
    raise NotImplementedError("Implement RRR-2003 pooling + negative-variance correction.")
