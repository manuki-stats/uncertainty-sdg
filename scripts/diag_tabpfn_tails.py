"""How often does TabPFN generate non-finite or wildly extreme values?

The r = 2 benchmark produced a file with y = inf. This fits TabPFN once at
n_org and generates many files, counting per file and column
  - non-finite values (NaN, +-inf), and
  - extreme values: more than 10 real-data SDs from the real-data mean
and printing where the extremes sit (both tails or one).

  python scripts/diag_tabpfn_tails.py --n-org 5000 --r 2 --n-files 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.simulation import SimulationDGP  # noqa: E402
from src.harness.run import make_generator_by_name  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-org", type=int, default=5000)
ap.add_argument("--r", type=int, default=2)
ap.add_argument("--n-files", type=int, default=20)
ap.add_argument("--kwargs", default="{}")
a = ap.parse_args()

import json  # noqa: E402

real = SimulationDGP().sample(a.n_org, np.random.default_rng(0))
mu, sd = real.mean(), real.std()
g = make_generator_by_name("tabpfn", json.loads(a.kwargs)).fit(real, train_seed=11)
n_syn = a.r * a.n_org
print(f"n_org {a.n_org}, r {a.r}: {a.n_files} files of {n_syn} rows")
print(f"{'seed':>4} {'sec':>5}  " + "  ".join(f"{c:>18}" for c in real.columns) + "   y range")

tot_rows = tot_bad = files_bad = 0
tails = {"low": 0, "high": 0}
for s in range(1, a.n_files + 1):
    t0 = time.perf_counter()
    f = g.generate(n_syn, gen_seed=s)
    secs = time.perf_counter() - t0
    cells, any_bad = [], False
    for c in real.columns:
        v = f[c].to_numpy(dtype=float)
        nonfin = int((~np.isfinite(v)).sum())
        fin = v[np.isfinite(v)]
        z = (fin - mu[c]) / sd[c]
        ext = int((np.abs(z) > 10).sum())
        tails["low"] += int((z < -10).sum()) + int((v == -np.inf).sum())
        tails["high"] += int((z > 10).sum()) + int((v == np.inf).sum())
        cells.append(f"nonfin {nonfin:3d} ext {ext:3d}")
        any_bad |= bool(nonfin or ext)
        tot_bad += nonfin + ext
    y = f["y"].to_numpy(dtype=float)
    yf = y[np.isfinite(y)]
    print(f"{s:4d} {secs:5.1f}  " + "  ".join(f"{x:>18}" for x in cells)
          + f"   [{yf.min():.3g}, {yf.max():.3g}]")
    tot_rows += n_syn
    files_bad += any_bad

print(f"\nfiles with any non-finite or extreme value: {files_bad}/{a.n_files}")
print(f"affected cells: {tot_bad} of {tot_rows * real.shape[1]} "
      f"({tot_bad / tot_rows:.2e} per row); low tail {tails['low']}, high tail {tails['high']}")
