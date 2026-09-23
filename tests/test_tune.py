import numpy as np

from src.generators.oracle import ParametricOracle
from src.harness.tune import TuneConfig, edges, run_tuning, sample_params

SPACE = {"epochs": {"choice": [300, 600, 900]},
         "lr": {"loguniform": [1e-4, 1e-2]},
         "dims": {"layers": {"width": [64, 128], "depth": [2, 3]}}}


def test_sampling_is_deterministic_and_in_range():
    a = [sample_params(SPACE, np.random.default_rng(3)) for _ in range(2)]
    assert a[0] == a[1]
    rng = np.random.default_rng(4)
    for _ in range(200):
        p = sample_params(SPACE, rng)
        assert p["epochs"] in (300, 600, 900)
        assert 1e-4 <= p["lr"] <= 1e-2
        assert len(p["dims"]) in (2, 3) and len(set(p["dims"])) == 1


def test_edges_flags_the_epoch_cap():
    assert any("epochs" in e for e in edges({"epochs": 900, "lr": 1e-3}, SPACE))


class ShrinkGen:
    """Correct plug-in model with y deviations scaled by `shrink`; the default
    0.6 under-disperses like TVAE, so the best setting (1.0) is known."""
    fits = 0

    def __init__(self, params=None):
        self.shrink = (params or {}).get("shrink", 0.6)

    def fit(self, data, *, train_seed):
        ShrinkGen.fits += 1
        self._g = ParametricOracle("plugin").fit(data, train_seed=train_seed)
        return self

    def generate(self, n, *, gen_seed):
        s = self._g.generate(n, gen_seed=gen_seed)
        s["y"] = s["y"].mean() + self.shrink * (s["y"] - s["y"].mean())
        return s


def factory(name, kwargs):
    return ShrinkGen(kwargs.get("params"))


def test_tuning_finds_the_known_best_and_resumes(tmp_path):
    cfg = TuneConfig(name="toy", generator="toy", n_tune=800, n_random=6,
                     n_files=1, n_top=2, n_seeds=2,
                     space={"shrink": {"choice": [0.5, 0.8, 1.0]}})
    ShrinkGen.fits = 0
    s = run_tuning(cfg, factory=factory, root=tmp_path)
    assert s["winner_params"] == {"shrink": 1.0}
    assert s["margin_beyond_2se"]
    assert 0.5 < s["reference_plugin_oracle"] < 2.0
    n_fits = ShrinkGen.fits
    assert n_fits == 7 + 3 * 2
    run_tuning(cfg, factory=factory, root=tmp_path)       # everything on disk: no refits
    assert ShrinkGen.fits == n_fits


class Flaky(ShrinkGen):
    """Fails while `broken` is set, like a generator with a bad dependency."""
    broken = True

    def fit(self, data, *, train_seed):
        if Flaky.broken:
            raise RuntimeError("simulated failure")
        return super().fit(data, train_seed=train_seed)


def test_failed_fits_are_retried_and_never_produce_a_winner(tmp_path):
    import pytest
    cfg = TuneConfig(name="flaky", generator="toy", n_tune=500, n_random=2,
                     n_files=1, n_top=1, n_seeds=2,
                     space={"shrink": {"choice": [0.5, 1.0]}})
    fac = lambda name, kw: Flaky(kw.get("params"))
    Flaky.broken = True
    with pytest.raises(RuntimeError, match="no winner"):
        run_tuning(cfg, factory=fac, root=tmp_path)
    assert not list(tmp_path.glob("*/winner.yaml"))
    Flaky.broken = False
    s = run_tuning(cfg, factory=fac, root=tmp_path)       # same folder: failures retried
    assert np.isfinite(s["winner_score"])


def test_missing_package_stops_the_run(tmp_path):
    import pytest
    def fac(name, kw):
        raise ModuleNotFoundError("No module named 'ctgan'")
    cfg = TuneConfig(name="nopkg", generator="toy", n_tune=500, n_random=1,
                     n_files=1, n_top=1, n_seeds=1, space={"shrink": {"choice": [1.0]}})
    with pytest.raises(ModuleNotFoundError):
        run_tuning(cfg, factory=fac, root=tmp_path)


def test_one_se_rule_takes_the_cheapest_indistinguishable_finalist():
    from src.harness.tune import select_winner
    rows = [dict(trial=0, stage2_mean=80.0, stage2_se=5.0, fit_s=30.0),
            dict(trial=13, stage2_mean=76.0, stage2_se=6.0, fit_s=1250.0),   # best, costly
            dict(trial=5, stage2_mean=90.0, stage2_se=2.0, fit_s=10.0)]      # cheap but > 1 SE
    sel, best = select_winner(rows)
    assert best["trial"] == 13 and sel["trial"] == 0
    rows[0]["stage2_mean"] = 83.0                                            # now outside 1 SE
    sel, best = select_winner(rows)
    assert sel["trial"] == 13
