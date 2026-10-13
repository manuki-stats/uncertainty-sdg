"""Step 2 (national): build the pseudo-population and the disjoint tuning sample.

Usage (repo root, env synth2):  python scripts/build_acs_extract.py
Streams the ACS 2024 1-year national person file, applies the agreed universe and filters,
reserves a tuning sample with a fixed seed, and writes the remaining pool as the
pseudo-population from which each replication draws its own D_org. Deterministic.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from acs_common import (COLS, EDUC_BINS, FLAGS, LABELS, RENAME, URL, ZIP, read_universe,
                        recode)

OUT = Path("extracts") / "acs2024_us"
SEED = 20261013
N_TUNE, N_ORG = 5000, 5000

names, u, counts, state_max = read_universe()
print(f"\n{', '.join(names)}: {counts['all persons']:,} person records, {len(state_max)} states")
print("base universe:")
for stage, n in list(counts.items())[1:]:
    print(f"  after {stage:<38} {n:>10,}")

# --- allocation profile on the base universe ---
alloc = pd.DataFrame({v: u[f].eq(1) for v, f in FLAGS.items()})
print("\nallocation rate within base universe:")
print(alloc.mean().sort_values(ascending=False).round(3).to_string())
clean = ~alloc.any(axis=1)
print(f"median WAGP, clean vs allocated: {u.WAGP[clean].median():,.0f} vs "
      f"{u.WAGP[~clean].median():,.0f}; ADJINC values: {sorted(u.ADJINC.unique())}")

# --- top-codes are state-specific: the atom is each state's maximum WAGP ---
topcoded = u.WAGP.eq(u.ST.map(state_max))
share = topcoded.groupby(u.ST).mean()
print(f"\ntop-code atom share by state (base universe): min {share.min():.2%}, "
      f"median {share.median():.2%}, max {share.max():.2%}; "
      f"states below 0.1%: {int((share < 0.001).sum())}")

# --- remaining filters ---
mask = pd.Series(True, index=u.index)
filters = [{"step": s, "remaining": n} for s, n in counts.items()]
for label, m in [("full year (WKWN >= 50)", u.WKWN >= 50),
                 ("no allocation on the 8 variables", clean),
                 ("below the state's top code", ~topcoded)]:
    mask &= m
    filters.append({"step": label, "remaining": int(mask.sum())})
    print(f"  after {label:<38} {mask.sum():>10,}")

pool = recode(u[mask].sort_values(["SERIALNO", "SPORDER"])).reset_index(drop=True)
perm = np.random.default_rng(SEED).permutation(len(pool))
d_tune = pool.iloc[perm[:N_TUNE]].reset_index(drop=True)
population = pool.drop(index=perm[:N_TUNE]).reset_index(drop=True)
f = N_ORG / len(population)
print(f"\npool {len(pool):,} -> tuning sample {len(d_tune):,} + population {len(population):,}; "
      f"sampling fraction of a D_org: f = {f:.2%}")


def summary(d):
    s = {"n": len(d)}
    for v in ["age", "hours", "wage"]:
        s[v] = {"mean": round(d[v].mean(), 2), "sd": round(d[v].std(), 2),
                "share_mode": round(d[v].eq(d[v].mode().iloc[0]).mean(), 3)}
    lw = np.log(d.wage)
    s["log_wage"] = {"mean": round(lw.mean(), 4), "sd": round(lw.std(), 4),
                     "skew": round(lw.skew(), 3), "excess_kurt": round(lw.kurt(), 3)}
    for v in ["sex", "marital", "educ", "class_worker", "nativity"]:
        s[v] = d[v].value_counts(normalize=True).round(4).to_dict()
    return s


summ = {"population": summary(population), "d_tune": summary(d_tune)}
OUT.mkdir(parents=True, exist_ok=True)
pop_csv = population.to_csv(index=False)
population.to_csv(OUT / "population.csv.gz", index=False,
                  compression={"method": "gzip", "mtime": 0})
d_tune.to_csv(OUT / "d_tune.csv", index=False)
manifest = {
    "source_url": URL, "source_files": names,
    "zip_sha256": hashlib.sha256(ZIP.read_bytes()).hexdigest(),
    "population_csv_sha256": hashlib.sha256(pop_csv.encode()).hexdigest(),
    "seed": SEED, "n_tune": N_TUNE, "n_org": N_ORG, "population_size": len(population),
    "sampling_fraction": f, "adjinc_applied": False,
    "top_code": "per-state maximum WAGP removed", "filters": filters,
    "educ_bins_schl": {lab: [lo, hi] for lo, hi, lab in EDUC_BINS},
    "labels": {RENAME[k]: v for k, v in LABELS.items()}, "columns": COLS, "summary": summ,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"written to {OUT}")

print("\n                     population     d_tune")
for v in ["age", "hours", "wage"]:
    print(f"mean {v:<15}" + "".join(f"{summ[k][v]['mean']:>11,.1f}" for k in summ))
for key in ["mean", "sd", "skew", "excess_kurt"]:
    print(f"log wage {key:<11}" + "".join(f"{summ[k]['log_wage'][key]:>11.3f}" for k in summ))
print(f"share at 40 hours   " + "".join(f"{summ[k]['hours']['share_mode']:>11.3f}" for k in summ))
for v in ["sex", "marital", "educ", "class_worker", "nativity"]:
    for lev in sorted(summ["population"][v]):
        print(f"{v[:8]}={lev:<17}" + "".join(f"{summ[k][v].get(lev, 0):>11.3f}" for k in summ))
exp = (population.marital.value_counts(normalize=True) * N_ORG).round(0).astype(int).to_dict()
print(f"\nexpected marital counts in a D_org of {N_ORG}: {exp}")
