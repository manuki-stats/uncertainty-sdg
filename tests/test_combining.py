"""Frozen reference values for the fully-synthetic combining rules.

The expected numbers were computed independently in R (see thesis Ch. 3) and
are pinned here, so any drift in the variance code fails loudly.
"""
import numpy as np
import pytest

from src.combining_rules.fully_synthetic import combine_fully_synthetic

U = [0.0400, 0.0450, 0.0380, 0.0420, 0.0410]


def test_positive_total_variance():
    r = combine_fully_synthetic([2.40, 1.60, 2.90, 1.20, 2.05], U, m=5)
    assert r.estimate == pytest.approx(2.03)
    assert r.b_m == pytest.approx(0.442)
    assert r.u_bar == pytest.approx(0.0412)
    assert r.total_variance == pytest.approx(0.4892)
    assert r.df == pytest.approx(3.4027172071)
    assert r.ci_low == pytest.approx(-0.0540339414)
    assert r.ci_high == pytest.approx(4.1140339414)
    assert not r.negative_variance


def test_negative_variance_correction():
    r = combine_fully_synthetic(
        [2.10, 1.95, 2.30, 1.88, 2.05], U, m=5, n_syn=1000, n_org=1000
    )
    assert r.negative_variance
    assert r.total_variance == pytest.approx(0.0412)
    assert r.ci_low == pytest.approx(1.6581707610)
    assert r.ci_high == pytest.approx(2.4538292390)
    assert np.isinf(r.df)


def test_correction_needs_sample_sizes():
    with pytest.raises(ValueError):
        combine_fully_synthetic([2.10, 1.95, 2.30, 1.88, 2.05], U, m=5)