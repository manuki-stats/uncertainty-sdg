"""The metrics: coverage, CI width, bias --- plus RMSE as the deliberate foil.

Across R replications:
    coverage = fraction of intervals containing the truth  (target 0.95)
    width    = mean interval length
    bias     = mean(pooled estimate) - truth
RMSE is computed too, and reported precisely to show it can look fine while
coverage fails --- the misleading foil, not a measure of inferential validity.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def coverage(intervals, truth: float) -> float:
    """Fraction of PooledInference objects whose interval contains the truth.

    Failed replications (NaN bounds) are dropped, not counted as misses, so a
    generator that occasionally collapses is not silently punished twice; the
    harness logs the failure count separately.
    """
    hits = [r.covers(truth) for r in intervals if np.isfinite(r.ci_low) and np.isfinite(r.ci_high)]
    return float(np.mean(hits)) if hits else float("nan")


def mean_width(intervals) -> float:
    w = [r.width for r in intervals if np.isfinite(r.width)]
    return float(np.mean(w)) if w else float("nan")


def bias(estimates: Sequence[float], truth: float) -> float:
    q = np.asarray(estimates, dtype=float)
    q = q[np.isfinite(q)]
    return float(q.mean() - truth) if q.size else float("nan")


def rmse_foil(estimates: Sequence[float], truth: float) -> float:
    """Reported as the foil, never as evidence of valid inference."""
    q = np.asarray(estimates, dtype=float)
    q = q[np.isfinite(q)]
    return float(np.sqrt(np.mean((q - truth) ** 2))) if q.size else float("nan")


def mc_standard_error(rate: float, n_reps: int) -> float:
    """Monte-Carlo standard error of an estimated coverage rate.

    At R = 500 and a true rate of 0.95 this is 0.010, so a gate band of
    [0.93, 0.97] is roughly two standard errors wide.
    """
    return float(np.sqrt(rate * (1.0 - rate) / n_reps))