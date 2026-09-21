"""The fast estimand path must agree with the statsmodels route exactly.

estimands_fast() skips patsy for speed, so this test is what licenses using it
for every number in the thesis.
"""
import numpy as np
import pandas as pd

from src.data.simulation import FORMULA, FOCAL, MAIN_TERM, HARD_TERM, SimulationDGP
from src.evaluation.estimands import e1_mean, e2_coefficient, e3_hard, estimands_fast


def test_fast_matches_statsmodels():
    df = SimulationDGP().sample(5000, np.random.default_rng(7))

    q_slow = np.array([
        e1_mean(df, FOCAL)[0],
        e2_coefficient(df, FORMULA, MAIN_TERM)[0],
        e3_hard(df, FORMULA, HARD_TERM)[0],
    ])
    u_slow = np.array([
        e1_mean(df, FOCAL)[1],
        e2_coefficient(df, FORMULA, MAIN_TERM)[1],
        e3_hard(df, FORMULA, HARD_TERM)[1],
    ])

    q_fast, u_fast = estimands_fast(df)
    np.testing.assert_allclose(q_fast, q_slow, rtol=0, atol=1e-12)
    np.testing.assert_allclose(u_fast, u_slow, rtol=0, atol=1e-15)