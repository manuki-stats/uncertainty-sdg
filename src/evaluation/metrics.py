"""The metrics: coverage, CI width, bias --- plus RMSE as the deliberate foil.

Across R replications:
    coverage = fraction of intervals containing the truth  (target 0.95)
    width    = mean interval length
    bias     = mean(pooled estimate) - truth
RMSE is computed too, and reported precisely to show it can look fine while
coverage fails --- the misleading foil, not a measure of inferential validity.
"""
from __future__ import annotations


def coverage(intervals, truth) -> float:
    raise NotImplementedError

def mean_width(intervals) -> float:
    raise NotImplementedError

def bias(estimates, truth) -> float:
    raise NotImplementedError

def rmse_foil(estimates, truth) -> float:
    """Reported as the foil, never as evidence of valid inference."""
    raise NotImplementedError
