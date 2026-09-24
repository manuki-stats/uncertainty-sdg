"""The TabPFN wrapper's own logic (column order, seeding, output types), tested
against stand-in modules, since torch and TabPFN are not installed everywhere.
Behaviour of TabPFN itself is checked by the pre-flight, not here."""
import sys
import types

import numpy as np
import pytest

from src.data.simulation import SimulationDGP


class _Est:
    def __init__(self, **kw):
        self.kw = kw


class _Unsup:
    last = None

    def __init__(self, tabpfn_clf, tabpfn_reg):
        self.clf, self.reg = tabpfn_clf, tabpfn_reg
        _Unsup.last = self

    def set_categorical_features(self, idx):
        self.cat = idx

    def fit(self, X):
        self.X_ = X
        return self

    def generate_synthetic_data(self, n_samples, t, n_permutations):
        rows = np.random.default_rng(np.random.randint(1 << 30)).integers(0, len(self.X_), n_samples)
        return self.X_[rows]                       # resample context rows: order preserved


@pytest.fixture
def fake_tabpfn(monkeypatch):
    tabpfn = types.ModuleType("tabpfn")
    tabpfn.TabPFNClassifier = tabpfn.TabPFNRegressor = _Est
    ext = types.ModuleType("tabpfn_extensions")
    uns = types.ModuleType("tabpfn_extensions.unsupervised")
    uns.TabPFNUnsupervisedModel = _Unsup
    for name, mod in {"tabpfn": tabpfn, "tabpfn_extensions": ext,
                      "tabpfn_extensions.unsupervised": uns}.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_order_seeds_and_output(fake_tabpfn):
    from src.generators.tabpfn_generator import TabPFNGenerator
    d = SimulationDGP().sample(300, np.random.default_rng(0))
    g = TabPFNGenerator().fit(d, train_seed=7)
    m = _Unsup.last
    assert m.clf.kw["random_state"] == 7 and m.reg.kw["ignore_pretraining_limits"]
    assert m.cat == [0]                              # x2 first in the visit order
    np.testing.assert_allclose(m.X_[:, 2], d["y"].to_numpy(np.float32))
    a, b, c = (g.generate(200, gen_seed=s) for s in (1, 1, 2))
    assert list(a.columns) == ["y", "x1", "x2"] and all(a.dtypes == float)
    assert a.equals(b) and not a.equals(c)
    assert set(np.unique(a["x2"])) <= {0.0, 1.0}


def test_rejects_badly_coded_discrete_column(fake_tabpfn):
    from src.generators.tabpfn_generator import TabPFNGenerator
    d = SimulationDGP().sample(100, np.random.default_rng(1))
    d["x2"] = d["x2"] + 1                            # coded 1/2
    with pytest.raises(ValueError, match="0..K-1"):
        TabPFNGenerator().fit(d, train_seed=1)
