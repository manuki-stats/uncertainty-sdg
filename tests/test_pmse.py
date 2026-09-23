import numpy as np
import pandas as pd

from src.data.simulation import FORMULA, SimulationDGP
from src.evaluation.pmse import MomentFeatures, pmse_score
from src.generators.oracle import ParametricOracle

DGP = SimulationDGP()


def test_regressors_are_the_ten_sufficient_statistic_monomials():
    d = DGP.sample(2000, np.random.default_rng(0))
    f = MomentFeatures(FORMULA).fit(d)
    assert set(f.names) == {"x1", "x2", "x1:x2", "x1*x1", "x1*x1:x2",
                            "y", "x1*y", "x2*y", "x1:x2*y", "y*y"}
    assert f.k == 11


def _mean_ratio(make_syn, reps=20, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(reps):
        orig = DGP.sample(n, rng)
        out.append(pmse_score(orig, make_syn(orig, i, rng), MomentFeatures(FORMULA).fit(orig)).ratio)
    return float(np.mean(out))


def test_correct_plugin_fit_scores_near_one():
    r = _mean_ratio(lambda o, i, rng: ParametricOracle("plugin").fit(o, train_seed=i)
                    .generate(len(o), gen_seed=i))
    assert 0.8 < r < 1.4


def test_detects_the_two_known_deep_generator_failures():
    no_interaction = SimulationDGP(b3=0.0)
    r_int = _mean_ratio(lambda o, i, rng: no_interaction.sample(len(o), rng))
    def shrunk(o, i, rng):
        s = DGP.sample(len(o), rng)
        s["y"] = s["y"].mean() + np.sqrt(0.72) * (s["y"] - s["y"].mean())
        return s
    r_disp = _mean_ratio(shrunk)
    assert r_int > 10 and r_disp > 5


def test_non_finite_synthetic_file_scores_infinite():
    d = DGP.sample(500, np.random.default_rng(2))
    bad = d.copy()
    bad.loc[0, "y"] = np.nan
    assert np.isinf(pmse_score(d, bad, MomentFeatures(FORMULA).fit(d)).ratio)
