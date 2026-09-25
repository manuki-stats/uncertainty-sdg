"""Does CTGAN's log_frequency setting matter here? A pre-specified, paired check.

ctgan's default log_frequency=True computes category probabilities from
log(count + 1) and sample() draws discrete columns from those probabilities,
which pulls every categorical marginal toward uniform (x2: 0.40 -> about 0.48).
This fits the tuned winner N times with fixed training seeds 101..100+N, with
the flag as given, and reports per fit and on average: the x2 share, E1-E3
against the real sample (as z-scores), and the pMSE ratio. Run it once with
true and once with false: the same seeds make the two runs a paired comparison.

  python scripts/diag_ctgan_logfreq.py --log-frequency false --n-fits 5 --gpu
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.simulation import FORMULA, SimulationDGP  # noqa: E402
from src.evaluation.estimands import estimands_fast  # noqa: E402
from src.evaluation.pmse import MomentFeatures, pmse_score  # noqa: E402
from src.harness.run import make_generator_by_name  # noqa: E402

WINNER = "results/tuning/tune_ctgan__350ba10a74/winner.yaml"

ap = argparse.ArgumentParser()
ap.add_argument("--log-frequency", choices=["true", "false"], required=True)
ap.add_argument("--n-fits", type=int, default=5)
ap.add_argument("--n-org", type=int, default=5000)
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args()

kwargs = yaml.safe_load(open(WINNER))["generator_kwargs"]
kwargs["params"] = {**kwargs["params"], "log_frequency": a.log_frequency == "true"}
kwargs["enable_gpu"] = a.gpu
print(f"ctgan {kwargs}")

real = SimulationDGP().sample(a.n_org, np.random.default_rng(0))
q_real, u_real = estimands_fast(real)
feats = MomentFeatures(FORMULA).fit(real)
print(f"real: x2 share {real.x2.mean():.3f}   E1-E3 {np.round(q_real, 3)}")
print(f"{'seed':>4} {'fit s':>6} {'x2 share':>9} {'E1 z':>6} {'E2 z':>6} {'E3 z':>6} {'pMSE':>7}")

rows = []
for s in range(101, 101 + a.n_fits):
    t0 = time.perf_counter()
    g = make_generator_by_name("ctgan", kwargs).fit(real, train_seed=s)
    secs = time.perf_counter() - t0
    f = g.generate(a.n_org, gen_seed=1)
    q, _ = estimands_fast(f)
    z = (q - q_real) / np.sqrt(2 * u_real)       # one file vs the real sample, r = 1
    pm = pmse_score(real, f, feats).ratio
    rows.append([f.x2.mean(), *z, pm])
    print(f"{s:4d} {secs:6.0f} {f.x2.mean():9.3f} {z[0]:+6.1f} {z[1]:+6.1f} {z[2]:+6.1f} {pm:7.1f}")

m = np.array(rows)
print(f"mean      {m[:, 0].mean():9.3f} {m[:, 1].mean():+6.1f} {m[:, 2].mean():+6.1f}"
      f" {m[:, 3].mean():+6.1f} {m[:, 4].mean():7.1f}")
print(f"sd        {m[:, 0].std(ddof=1):9.3f} {m[:, 1].std(ddof=1):6.1f} {m[:, 2].std(ddof=1):6.1f}"
      f" {m[:, 3].std(ddof=1):6.1f} {m[:, 4].std(ddof=1):7.1f}")
print(f"mean |E1 z| {np.abs(m[:, 1]).mean():.1f}")
