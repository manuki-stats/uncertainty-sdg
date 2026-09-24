"""Cell-level metrics computed from replication-level results.

Coverage, width, bias and the variance ratio are computed directly in
summarise.py from the raw chunks; this module holds the pieces that need care.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats


def wilson_interval(hits: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (coverage out of R)."""
    if n == 0:
        return float("nan"), float("nan")
    z = stats.norm.ppf(0.5 + level / 2)
    p = hits / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)


def needs_escalation(hits: int, n: int, threshold: float = 0.93, r_max: int = 1000) -> bool:
    """Two-pass rule: rerun at r_max if near-nominal coverage cannot be ruled out."""
    return n < r_max and wilson_interval(hits, n)[1] >= threshold


def mc_standard_error(rate: float, n_reps: int) -> float:
    """Monte Carlo standard error of an estimated coverage rate."""
    return float(np.sqrt(rate * (1.0 - rate) / n_reps))


def rmse_foil(estimates: Sequence[float], truth) -> float:
    """Estimate-level RMSE of the pooled point estimate around the truth, across
    replications. Reported as the foil, never as evidence of valid inference: a
    generator can post a small RMSE with intervals that miss. Record-level
    reconstruction error is undefined under full synthesis, since no synthetic
    record corresponds to an original one."""
    err = np.asarray(estimates, dtype=float) - np.asarray(truth, dtype=float)
    err = err[np.isfinite(err)]
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
