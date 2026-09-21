"""Frozen reference values for the combining rules.

Recompute them independently in R before trusting them. Any drift in the variance code then fails loudly.
"""
import numpy as np
import pytest

from src.combining_rules.rules import naive, pool

Q = [2.40, 1.60, 2.90, 1.20, 2.05]
U = [0.0400, 0.0450, 0.0380, 0.0420, 0.0410]
R_RATIO = 2


def test_components_and_kappa():
    p = pool(Q, U, r=R_RATIO)
    assert p.q_bar == pytest.approx(2.03)
    assert p.b == pytest.approx(0.442)
    assert p.u_bar == pytest.approx(0.0412)
    assert p.kappa == pytest.approx(0.442 / 0.0412)


def test_population_estimators():
    iv = pool(Q, U, r=R_RATIO).intervals
    assert iv["T_alt"].variance == pytest.approx(2 * 0.0412 + 0.442 / 5)
    assert iv["T_alt"].df == pytest.approx(14.932454290452696)
    assert iv["T_alt"].lo == pytest.approx(1.148768281551511)
    assert iv["T_s"].variance == pytest.approx((2 + 1 / 5) * 0.0412)
    assert np.isinf(iv["T_s"].df)
    assert iv["T_s_ppd"].variance == pytest.approx((2 + 3 / 5) * 0.0412)
    assert iv["T_f"].df == pytest.approx(3.4027172070823917)   # same as the old T_f test


def test_sample_target_drops_sampling_term():
    iv = pool(Q, U, r=R_RATIO, target="sample").intervals
    assert iv["T_alt"].variance == pytest.approx(0.442 / 5)
    assert iv["T_alt"].df == 4
    assert iv["T_s"].variance == pytest.approx(0.0412 / 5)
    assert iv["T_s_ppd"].variance == pytest.approx(3 * 0.0412 / 5)
    assert "T_f" not in iv


def test_T_p_is_T_alt_at_r_equal_one():
    iv = pool(Q, U, r=1).intervals["T_alt"]
    assert iv.variance == pytest.approx(0.0412 + 0.442 / 5)


def test_T_f_correction_fires_on_negative_variance():
    iv = pool([2.10, 1.95, 2.30, 1.88, 2.05], U, r=R_RATIO).intervals["T_f"]
    assert iv.corrected
    assert iv.variance == pytest.approx(R_RATIO * 0.0412)
    assert np.isinf(iv.df)


def test_naive_and_guards():
    iv = naive(2.0, 0.04)
    assert iv.lo == pytest.approx(2.0 - 1.959963984540054 * 0.2)
    with pytest.raises(ValueError):
        pool([1.0], [0.1], r=2)
    assert pool([1.0, np.nan], [0.1, 0.1], r=2).intervals == {}
