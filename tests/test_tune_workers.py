"""Parallel tuning workers (slurm/tune.sbatch) reproduce a sequential run.

Workers are separate processes, as in production: SciPy's L-BFGS-B, which the
pMSE fit uses, is not thread-safe in older SciPy releases, so threads would test
something production never does.
"""
import json
import multiprocessing as mp
import os

import numpy as np

from src.data.acs import FEATURE_FORMULA
from src.harness.tune import TuneConfig, _fit_path, run_tuning


class _Noisy:
    """Resampler with tunable noise on log_wage, seeded by both seeds. Appends
    one line per fit to $FIT_LOG, so the test can count fits across processes."""
    def __init__(self, params):
        self.s = float(params.get("noise", 0.0))

    def fit(self, data, *, train_seed):
        if os.environ.get("FIT_LOG"):
            with open(os.environ["FIT_LOG"], "a") as f:
                f.write(f"{train_seed}\n")
        self.d, self.train_seed = data, train_seed
        return self

    def generate(self, n, *, gen_seed):
        rng = np.random.default_rng([gen_seed, self.train_seed])
        out = self.d.iloc[rng.integers(len(self.d), size=n)].reset_index(drop=True)
        out["log_wage"] = out["log_wage"] + rng.normal(0, self.s, n)
        return out


def _factory(generator, kwargs):
    return _Noisy(kwargs["params"])


CFG = TuneConfig(name="tw", generator="noisy", dataset="acs", formula=FEATURE_FORMULA,
                 n_tune=1000, n_random=7, n_files=2, n_top=2, n_seeds=2,
                 space={"noise": {"choice": [0.0, 0.1, 0.3, 0.6]},
                        "pad": {"loguniform": [1.0, 10.0]}})


def _worker(args):
    root, stage = args
    run_tuning(CFG, _factory, root=root, worker_stage=stage)


def _fits(root):
    out = next(root.glob("tw__*"))
    return {p.name: json.loads(p.read_text()) for p in sorted(out.glob("fit_*.json"))}


def _n_logged(path):
    return len(path.read_text().splitlines()) if path.exists() else 0


def test_parallel_workers_equal_sequential(tmp_path, monkeypatch):
    seq = run_tuning(CFG, _factory, root=tmp_path / "seq")
    log = tmp_path / "fits.log"
    monkeypatch.setenv("FIT_LOG", str(log))            # inherited by the spawned workers
    with mp.get_context("spawn").Pool(4) as pool:
        for stage in (1, 2):
            pool.map(_worker, [(tmp_path / "par", stage)] * 4)
    n_parallel = _n_logged(log)
    par = run_tuning(CFG, _factory, root=tmp_path / "par", retry_failed=False)

    assert n_parallel == (CFG.n_random + 1) + (CFG.n_top + 1) * CFG.n_seeds   # each fit once
    assert _n_logged(log) == n_parallel                                       # final pass: none
    a, b = _fits(tmp_path / "seq"), _fits(tmp_path / "par")
    assert a.keys() == b.keys()
    for k in a:
        np.testing.assert_allclose(a[k]["ratios"], b[k]["ratios"], rtol=1e-4)
    # Same finalists, scores and best-by-mean. The 1-SE winner can still differ:
    # it is the cheapest by MEASURED fit time (here every toy fit costs the same).
    key = lambda s: [r["trial"] for r in s["stage2_table"]]
    assert key(par) == key(seq)
    np.testing.assert_allclose([r["stage2_mean"] for r in par["stage2_table"]],
                               [r["stage2_mean"] for r in seq["stage2_table"]], rtol=1e-4)
    assert par["best_by_mean_trial"] == seq["best_by_mean_trial"]


def test_claims_stale_and_live(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "thisjob")
    out = tmp_path / f"tw__{CFG.hash()}"
    out.mkdir(parents=True)
    _fit_path(out, "stage1", 0, 0).with_suffix(".claim").write_text("oldjob:1")   # stale
    _fit_path(out, "stage1", 1, 0).with_suffix(".claim").write_text("thisjob:9")  # live
    run_tuning(CFG, _factory, root=tmp_path, worker_stage=1)
    assert _fit_path(out, "stage1", 0, 0).exists()          # stale claim taken over
    assert not _fit_path(out, "stage1", 1, 0).exists()      # live claim respected
    run_tuning(CFG, _factory, root=tmp_path, worker_stage=2)   # stage 1 incomplete: exits
    assert not list(out.glob("fit_stage2_*.json"))
    s = run_tuning(CFG, _factory, root=tmp_path)             # final pass fills the gap
    assert _fit_path(out, "stage1", 1, 0).exists() and s["failed_fits"] == 0


def test_pmse_is_reproducible_under_round_off():
    """A round-off-sized change in the input must not move the pMSE ratio: the
    propensity fit has to converge, not stop on a relative-decrease test."""
    from src.data.acs import derive, load_tuning_sample
    from src.evaluation.pmse import DerivedMomentFeatures, pmse_score

    t = load_tuning_sample()
    f = DerivedMomentFeatures(FEATURE_FORMULA, derive).fit(t)
    rng = np.random.default_rng(2)
    for noise in (0.0, 0.3):
        s = t.iloc[rng.integers(len(t), size=len(t))].reset_index(drop=True)
        s["log_wage"] = s["log_wage"] + rng.normal(0, noise, len(s))
        s2 = s.copy()
        s2["log_wage"] = s2["log_wage"] * (1 + 1e-14 * rng.standard_normal(len(s2)))
        a, b = pmse_score(t, s, f).ratio, pmse_score(t, s2, f).ratio
        assert abs(a - b) / a < 1e-4
