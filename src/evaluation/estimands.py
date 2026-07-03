"""The estimand battery E1-E3, computed on each synthetic dataset (cheap).

    E1  univariate mean of a focal variable   --- fit litmus (determinism can't bias a mean)
    E2  coefficient of a main predictor        --- undercoverage
    E3  hard estimand (interaction / low-density) --- bias + undercoverage

Each returns a point estimate and its within-dataset variance (q, u), the two
inputs the combining rules need.
"""
from __future__ import annotations
import pandas as pd


def e1_mean(df: pd.DataFrame, var: str):
    """Point estimate and variance of a univariate mean. TO IMPLEMENT."""
    raise NotImplementedError

def e2_coefficient(df: pd.DataFrame, formula: str, term: str):
    """A main-predictor regression coefficient and its variance. TO IMPLEMENT."""
    raise NotImplementedError

def e3_hard(df: pd.DataFrame, formula: str, term: str):
    """An interaction / low-density coefficient and its variance. TO IMPLEMENT."""
    raise NotImplementedError
