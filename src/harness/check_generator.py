"""Pre-flight checks for a generator wrapper, before any coverage number.

  1. runs      fit + generate return the expected columns as floats, x2 in {0,1}
  2. seeds     same seeds -> identical file; new gen seed -> different file from
               the same fit. Whether a new TRAIN seed changes the fit is reported,
               not judged: it should for neural nets and proper synthpop, and
               should NOT for a deterministic fit (MLE oracle, synthpop proper=FALSE)
  3. dispersion  Var(synthetic col) / Var(real col), should sit near 1
  4. timing    seconds per fit and per file at the requested n_org and r; the
               projection uses the second (steady-state) fit and file, serial
  5. moments   which of the propensity model's ten moments the synthetic files
               miss (z-scores of the mean difference, worst first), and the
               pMSE ratio: the "why" behind a bad tuning score

Usage:
  python -m src.harness.check_generator --generator ctgan --n-org 2000 --r 2
  python -m src.harness.check_generator --generator synthpop --kwargs '{"proper": true}'
  python -m src.harness.check_generator --winner results/tuning/<run>/winner.yaml --n-org 5000
  python -m src.harness.check_generator --winner <...>/winner.yaml --kwargs '{"enable_gpu": true}'
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.data.simulation import FORMULA, SimulationDGP
from src.evaluation.estimands import estimands_fast
from src.evaluation.pmse import MomentFeatures, pmse_score
from src.harness.run import make_generator_by_name

EXPECTED = ["y", "x1", "x2"]


def check(name: str, kwargs: dict, n_org: int, r: int) -> None:
    n_syn = r * n_org
    real = SimulationDGP().sample(n_org, np.random.default_rng(0))
    ok = lambda flag: "OK " if flag else "FAIL"

    # 1. runs
    t0 = time.perf_counter()
    g = make_generator_by_name(name, kwargs).fit(real, train_seed=11)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    f1 = g.generate(n_syn, gen_seed=1)
    t_gen = time.perf_counter() - t0
    cols_ok = list(f1.columns) == EXPECTED
    float_ok = all(pd.api.types.is_float_dtype(f1[c]) for c in EXPECTED)
    x2_ok = set(np.unique(f1["x2"])) <= {0.0, 1.0}
    shape_ok = len(f1) == n_syn and f1.isna().sum().sum() == 0
    print(f"[1 runs]      columns {ok(cols_ok)}  floats {ok(float_ok)}  "
          f"x2 in {{0,1}} {ok(x2_ok)}  shape/NaN {ok(shape_ok)}")
    q, u = estimands_fast(f1)
    q_real, u_real = estimands_fast(real)
    # Expected sd of ONE file's estimate around the real one under proper
    # synthesis: posterior noise (u_real) plus record noise (u_real / r).
    z = (q - q_real) / np.sqrt(u_real * (1 + 1 / r))
    print(f"              E1-E3 synthetic: {np.round(q, 3)}   real: {np.round(q_real, 3)}")
    print(f"              E1 litmus z={z[0]:+.1f} {'OK' if abs(z[0]) <= 3 else 'FAIL: mean not reproduced'}"
          f"     E2/E3 misfit z={z[1]:+.1f}, {z[2]:+.1f}"
          f"{'' if max(abs(z[1]), abs(z[2])) <= 3 else '  <- relationship distorted'}")

    def slopes(df):
        return [np.polyfit(df.x1[df.x2 == g], df.y[df.x2 == g], 1)[0] for g in (0.0, 1.0)]
    sr, ss = slopes(real), slopes(f1)
    print(f"              slope of y on x1 | x2=0: real {sr[0]:.2f} syn {ss[0]:.2f}"
          f"   | x2=1: real {sr[1]:.2f} syn {ss[1]:.2f}   (DGP: 2.0 and 3.5)")

    # 2. seeds
    f1_again = g.generate(n_syn, gen_seed=1)
    t0 = time.perf_counter()
    f2 = g.generate(n_syn, gen_seed=2)
    t_gen2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    g2 = make_generator_by_name(name, kwargs).fit(real, train_seed=12)
    t_fit2 = time.perf_counter() - t0
    f3 = g2.generate(n_syn, gen_seed=1)
    same = f1.equals(f1_again)
    diff_gen = not f1.equals(f2)
    diff_train = not f1.equals(f3)
    print(f"[2 seeds]     same seeds identical {ok(same)}   new gen seed differs {ok(diff_gen)}   "
          f"new train seed changes fit: {'yes' if diff_train else 'no'} "
          f"(expected yes for neural nets / proper synthpop, no for a deterministic fit)")
    if not same:
        print("              -> NOT deterministic. On GPU try torch.use_deterministic_algorithms(True);"
              " on CPU this is a wrapper bug.")

    # 3. dispersion
    files = [f1, f2, g.generate(n_syn, gen_seed=3)]
    ratio = pd.concat([f.var() / real.var() for f in files], axis=1).mean(axis=1)
    means = pd.concat([f.mean() - real.mean() for f in files], axis=1).mean(axis=1)
    print("[3 dispersion] Var(syn)/Var(real):  " +
          "  ".join(f"{c}={v:.3f}" for c, v in ratio.items()) +
          "     mean(syn)-mean(real):  " + "  ".join(f"{c}={v:+.3f}" for c, v in means.items()))
    if (ratio < 0.8).any() or (ratio > 1.25).any():
        print("              -> dispersion off by >20% in some column; investigate before running.")

    # 4. timing
    # The first fit and file in a process also pay one-off costs (loading
    # weights, CUDA start-up, TabPFN's warm-up), paid once per job, not per fit.
    # Project from the second fit and file.
    print(f"[4 timing]    n_org={n_org} r={r}:  first fit {t_fit:.1f}s, first file {t_gen:.2f}s"
          f"  (incl. one-off start-up)   steady: fit {t_fit2:.1f}s, file {t_gen2:.2f}s")
    for m, K, R in ((10, 10, 100), (10, 10, 1000)):
        steps12 = R * (t_fit2 + m * t_gen2)
        step34 = R * K * (t_fit2 + t_gen2)
        print(f"              projected: Steps 1-2 at m={m}, R={R}: {steps12/3600:6.1f} h   "
              f"one ensemble step at K={K}, R={R}: {step34/3600:6.1f} h")

    # 5. moments
    feats = MomentFeatures(FORMULA).fit(real)
    Fr = feats.transform(real)
    zs = []
    for f in files:
        Fs = feats.transform(f)
        se = np.sqrt(Fr.var(axis=0, ddof=1) / len(Fr) + Fs.var(axis=0, ddof=1) / len(Fs))
        zs.append((Fs.mean(axis=0) - Fr.mean(axis=0)) / se)
    z = np.mean(zs, axis=0)
    order = np.argsort(-np.abs(z))
    # pMSE at r = 1 (first n_org rows), the scale used in tuning: the null
    # reference shifts with the synthetic sample size.
    ratios = [pmse_score(real, f.iloc[:n_org], feats).ratio for f in files]
    print(f"[5 moments]   pMSE ratio at r=1: {np.mean(ratios):.1f} (tuning scale; correct plug-in"
          f" fit about 1, proper about 2).  Moment mismatch z, worst first:")
    print("              " + "   ".join(f"{feats.names[j]} {z[j]:+.1f}" for j in order))
    bad = [feats.names[j] for j in order if abs(z[j]) > 4]
    if bad:
        print(f"              -> |z| > 4: {', '.join(bad)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator")
    ap.add_argument("--winner", help="winner.yaml from a tuning run (sets generator and kwargs)")
    ap.add_argument("--kwargs", default="{}", help="JSON, e.g. '{\"proper\": true}'")
    ap.add_argument("--n-org", type=int, default=2000)
    ap.add_argument("--r", type=int, default=2)
    a = ap.parse_args()
    if a.winner:
        import yaml
        w = yaml.safe_load(open(a.winner))
        # --kwargs on top of a winner overrides single keys, e.g. enable_gpu
        name, kwargs = w["generator"], {**w.get("generator_kwargs", {}), **json.loads(a.kwargs)}
        print(f"{a.winner}: {name} {kwargs}")
    elif a.generator:
        name, kwargs = a.generator, json.loads(a.kwargs)
    else:
        ap.error("give --generator or --winner")
    check(name, kwargs, a.n_org, a.r)


if __name__ == "__main__":
    main()
