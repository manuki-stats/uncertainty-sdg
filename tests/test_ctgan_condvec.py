"""The generation-only fix for ctgan's log-frequency condition vectors, tested
on a stand-in for ctgan's DataSampler (torch is not installed everywhere).
The real sampler is checked on the cluster by check_generator (x2 share)."""
import numpy as np

from src.generators.ctgan_generators import use_empirical_condvec


class _Sampler:
    """The two attributes use_empirical_condvec touches, laid out as ctgan
    0.12.1 lays them out: one row per discrete column, zero-padded."""

    def __init__(self, counts_per_col, log_frequency=True):
        width = max(len(c) for c in counts_per_col)
        self._rid_by_cat_cols = [[np.arange(k) for k in counts] for counts in counts_per_col]
        self._discrete_column_category_prob = np.zeros((len(counts_per_col), width))
        for i, counts in enumerate(counts_per_col):
            f = np.log(np.array(counts, float) + 1) if log_frequency else np.array(counts, float)
            self._discrete_column_category_prob[i, : len(counts)] = f / f.sum()


def test_binary_column_gets_empirical_share():
    s = _Sampler([[3000, 2000]])                       # x2: 60% zeros, 40% ones
    assert abs(s._discrete_column_category_prob[0, 1] - 0.487) < 0.001   # the bug
    use_empirical_condvec(s)
    np.testing.assert_allclose(s._discrete_column_category_prob[0], [0.6, 0.4])


def test_several_columns_padding_and_generation_draw():
    s = _Sampler([[900, 90, 10], [500, 500]])
    use_empirical_condvec(s)
    p = s._discrete_column_category_prob
    np.testing.assert_allclose(p[0], [0.9, 0.09, 0.01])
    np.testing.assert_allclose(p[1], [0.5, 0.5, 0.0])  # padding stays zero
    # sample_original_condvec: flatten, drop zeros, renormalise -> each column
    # is chosen with probability 1/2 and its categories by their real shares
    flat = p.flatten(); flat = flat[flat != 0]; flat = flat / flat.sum()
    np.testing.assert_allclose(flat[:3] / flat[:3].sum(), [0.9, 0.09, 0.01])
