"""The real-data path: coding, the M2 fast path, sampling, tuning features."""
import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from src.data.acs import (DISCRETE, EXTRACT, FEATURE_FORMULA, LEVELS, ORDER, ACSPopulation,
                          derive, estimands_acs, load_tuning_sample)
from src.evaluation.pmse import DerivedMomentFeatures, pmse_score


@pytest.fixture(scope="module")
def pop():
    return ACSPopulation()


def test_coded_frame(pop):
    d = pop.data
    assert tuple(d.columns) == ORDER
    assert all(d.dtypes == float)
    for c in DISCRETE:
        assert set(d[c].unique()) == set(float(i) for i in range(len(LEVELS[c])))
    assert np.all(np.isfinite(d.to_numpy()))


def test_truth_matches_build_script(pop):
    stored = json.loads((EXTRACT / "population_truth.json").read_text())["Q"]
    for k, v in pop.truth().items():
        assert v == pytest.approx(stored[k], abs=1e-9)


def test_fast_path_matches_statsmodels(pop):
    d = pop.sample(5000, np.random.default_rng(1))
    q, u = estimands_acs(d)
    fit = smf.ols(FEATURE_FORMULA, data=derive(d)).fit()
    y = d["log_wage"].to_numpy()
    np.testing.assert_allclose(q, [y.mean(), fit.params["a"], fit.params["a:female"]], rtol=1e-9)
    np.testing.assert_allclose(u, [y.var(ddof=1) / y.size, fit.bse["a"] ** 2,
                                   fit.bse["a:female"] ** 2], rtol=1e-9)


def test_unproduced_level_is_dropped_like_patsy(pop):
    d = pop.sample(5000, np.random.default_rng(2))
    d = d[d["educ"] != LEVELS["educ"].index("lt_hs")].reset_index(drop=True)
    q, u = estimands_acs(d)
    fit = smf.ols(FEATURE_FORMULA.replace("e_lt_hs + ", ""), data=derive(d)).fit()
    np.testing.assert_allclose(q[1:], [fit.params["a"], fit.params["a:female"]], rtol=1e-9)
    np.testing.assert_allclose(u[1:], [fit.bse["a"] ** 2, fit.bse["a:female"] ** 2], rtol=1e-9)


def test_degenerate_file_gives_nan_not_crash(pop):
    d = pop.sample(500, np.random.default_rng(3)).assign(sex=0.0)     # no women
    q, u = estimands_acs(d)
    assert np.isfinite(q[0]) and np.isnan(q[1:]).all() and np.isnan(u[1:]).all()


def test_sampling_reproducible(pop):
    a = pop.sample(5000, np.random.default_rng(7))
    b = pop.sample(5000, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 5000 and tuple(a.columns) == ORDER


def test_tuning_sample_and_features():
    t = load_tuning_sample()
    assert len(t) == 5000 and tuple(t.columns) == ORDER
    feats = DerivedMomentFeatures(FEATURE_FORMULA, derive).fit(t)
    assert 20 <= feats.k <= 60
    other = ACSPopulation().sample(5000, np.random.default_rng(11))
    r = pmse_score(t, other, feats)            # two independent real samples: near the null
    assert 0.3 < r.ratio < 3.0


# ---------------------------------------------------------------- harness wiring

class _Resampler:
    """Plug-in empirical 'generator' for wiring tests: resamples the training rows."""
    def fit(self, data, *, train_seed):
        self._d = data
        return self

    def generate(self, n, *, gen_seed):
        return self._d.sample(n=n, replace=True,
                              random_state=gen_seed % 2**32).reset_index(drop=True)


def test_sample_target_after_the_fact_equals_direct_run(tmp_path, monkeypatch):
    import src.harness.run as run_mod
    import src.results_io as rio
    from src.config import ExperimentConfig
    from src.evaluation.summarise import sample_target_rows

    monkeypatch.setattr(rio, "RAW_ROOT", tmp_path)
    monkeypatch.setattr(run_mod, "make_generator", lambda cfg: _Resampler())
    kw = dict(generator="resampler", dataset="acs", n_org=1000, r=2, m=3, K=3,
              estimators=("T_alt", "T_s", "T_s_ppd"))
    pop_cfg = ExperimentConfig(name="wiring_pop", target="population", **kw)
    smp_cfg = ExperimentConfig(name="wiring_smp", target="sample", **kw)
    run_mod.run(pop_cfg, 0, 3, 2)
    run_mod.run(smp_cfg, 0, 3, 2)

    keys = ["rep", "step", "estimand", "estimator"]
    after = sample_target_rows(pop_cfg).sort_values(keys).reset_index(drop=True)
    direct = rio.read_cell(smp_cfg.name, smp_cfg.hash()).sort_values(keys).reset_index(drop=True)
    assert len(after) == len(direct) > 0
    pd.testing.assert_frame_equal(after[keys], direct[keys])
    for c in ["target_value", "estimate", "variance", "lo", "hi"]:
        np.testing.assert_allclose(after[c], direct[c], rtol=1e-9, atol=1e-12)  # CSV round trip
    assert (after["covered"].to_numpy() == direct["covered"].to_numpy()).all()


def test_synthpop_input_frame_unchanged_for_simulation():
    """The generalised wrapper must hand R exactly what the simulation always did."""
    from src.data.simulation import SimulationDGP
    from src.generators.synthpop_generator import SynthpopGenerator

    sim = SimulationDGP().sample(300, np.random.default_rng(0))
    new = SynthpopGenerator(proper=False).fit(sim, train_seed=1)._data
    old = sim[["x2", "x1", "y"]].astype(float).reset_index(drop=True)
    old["x2"] = pd.Categorical(old["x2"].astype(int).astype(str), categories=["0", "1"])
    pd.testing.assert_frame_equal(new, old)


def test_synthpop_input_frame_acs(pop):
    from src.generators.synthpop_generator import SynthpopGenerator

    d = pop.sample(2000, np.random.default_rng(4))
    g = SynthpopGenerator(proper=False, order=ORDER, discrete_columns=DISCRETE).fit(d, train_seed=1)
    assert list(g._data.columns) == list(ORDER)
    for c in DISCRETE:
        assert isinstance(g._data[c].dtype, pd.CategoricalDtype)
        assert list(g._data[c].cat.categories) == [str(i) for i in range(len(LEVELS[c]))]


def test_tabpfn_categoricals_are_exactly_the_declared_columns(pop):
    """tabpfn_extensions infers extra categoricals (<= 40 distinct values); the wrapper
    must undo that, or age comes back as class indices 0-39."""
    pytest.importorskip("tabpfn_extensions")
    from src.generators.tabpfn_generator import TabPFNGenerator

    d = pop.sample(2000, np.random.default_rng(5))
    g = TabPFNGenerator(order=ORDER, discrete_columns=DISCRETE, warmup=False, device="cpu")
    g.fit(d, train_seed=1)
    assert sorted(g._model.categorical_features) == sorted(ORDER.index(c) for c in DISCRETE)
    assert "age" in g.inferred_categorical          # the case the reset exists for


def test_tabpfn_simulation_categoricals_unchanged():
    pytest.importorskip("tabpfn_extensions")
    from src.data.simulation import SimulationDGP
    from src.generators.tabpfn_generator import TabPFNGenerator

    sim = SimulationDGP().sample(2000, np.random.default_rng(0))
    g = TabPFNGenerator(warmup=False, device="cpu").fit(sim, train_seed=1)
    assert g._model.categorical_features == [0] and g.inferred_categorical == []
