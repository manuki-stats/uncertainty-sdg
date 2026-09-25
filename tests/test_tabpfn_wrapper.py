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

    inject_inf = False                               # set by the rejection test

    def generate_synthetic_data(self, n_samples, t, n_permutations):
        self.calls = getattr(self, "calls", []) + [n_samples]
        rows = np.random.default_rng(np.random.randint(1 << 30)).integers(0, len(self.X_), n_samples)
        out = self.X_[rows].copy()                  # resample context rows: order preserved
        if _Unsup.inject_inf and n_samples >= 100:  # full files only, never the redraws
            out[5, 2] = np.inf                      # y is last in the visit order
        return out


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
    import src.generators.tabpfn_generator as T
    monkeypatch.setattr(T, "_WARMED_UP", False)            # a fresh "process" per test


def test_order_seeds_and_output(fake_tabpfn):
    from src.generators.tabpfn_generator import TabPFNGenerator
    d = SimulationDGP().sample(300, np.random.default_rng(0))
    g = TabPFNGenerator().fit(d, train_seed=7)
    m = _Unsup.last
    assert m.clf.kw["random_state"] == 7 and m.reg.kw["ignore_pretraining_limits"]
    assert m.cat == [0]                              # x2 first in the visit order
    assert m.calls == [2]                            # warm-up happened inside fit()
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


def test_warm_up_once_per_process_and_cache_pinned(fake_tabpfn, monkeypatch):
    import os
    from src.generators.tabpfn_generator import TabPFNGenerator
    monkeypatch.setenv("TABPFN_MODEL_CACHE_SIZE", "0")
    d = SimulationDGP().sample(100, np.random.default_rng(2))
    TabPFNGenerator().fit(d, train_seed=1)
    first = _Unsup.last
    TabPFNGenerator().fit(d, train_seed=2)
    second = _Unsup.last
    assert first.calls == [2] and not hasattr(second, "calls")
    assert os.environ["TABPFN_MODEL_CACHE_SIZE"] == "2"


def test_non_finite_rows_are_rejected_and_redrawn(fake_tabpfn, monkeypatch):
    from src.generators.tabpfn_generator import TabPFNGenerator
    monkeypatch.setattr(_Unsup, "inject_inf", True)
    d = SimulationDGP().sample(300, np.random.default_rng(3))
    g = TabPFNGenerator().fit(d, train_seed=1)
    a = g.generate(200, gen_seed=4)
    assert len(a) == 200 and np.isfinite(a.to_numpy()).all()
    assert g.last_n_rejected == 1
    assert _Unsup.last.calls[-1] == 2              # one redraw call, of max(k, 2) rows
    assert a.equals(g.generate(200, gen_seed=4))   # still reproducible from the seed
    assert g.generate(200, gen_seed=5) is not None and g.last_n_rejected == 1
