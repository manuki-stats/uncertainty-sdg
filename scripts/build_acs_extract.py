"""Step 2 of the real-data extract: build D_org and the disjoint tuning sample from ACS 2024 PUMS.

Usage (repo root, env synth2):  python scripts/build_acs_extract.py
Reads the zip cached by acs_profile.py, applies the agreed universe, filters and recodes,
draws two disjoint samples from one fixed-seed permutation, and writes them to OUT with a
manifest. Rerunning reproduces the files exactly.
"""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

YEAR, ST = 2024, "ca"
URL = f"https://www2.census.gov/programs-surveys/acs/data/pums/{YEAR}/1-Year/csv_p{ST}.zip"
ZIP = Path("acs_raw") / f"acs{YEAR}_1y_csv_p{ST}.zip"
OUT = Path("extracts") / f"acs{YEAR}_{ST}"
SEED = 20261013
N_ORG, N_TUNE = 5000, 5000

VARS = ["AGEP", "WKHP", "WAGP", "SEX", "MAR", "SCHL", "COW", "NATIVITY"]
FLAGS = {"AGEP": "FAGEP", "WKHP": "FWKHP", "WAGP": "FWAGP", "SEX": "FSEXP", "MAR": "FMARP",
         "SCHL": "FSCHLP", "COW": "FCOWP", "NATIVITY": "FPOBP"}
LABELS = {
    "SEX": {1: "male", 2: "female"},
    "MAR": {1: "married", 2: "widowed", 3: "divorced", 4: "separated", 5: "never_married"},
    "COW": {1: "private_profit", 2: "private_nonprofit", 3: "local_gov", 4: "state_gov",
            5: "federal_gov"},
    "NATIVITY": {1: "native", 2: "foreign_born"},
}
EDUC_BINS = [(1, 15, "lt_hs"), (16, 17, "hs"), (18, 20, "some_college"), (21, 21, "bachelor"),
             (22, 24, "graduate")]
RENAME = {"AGEP": "age", "WKHP": "hours", "WAGP": "wage", "SEX": "sex", "MAR": "marital",
          "COW": "class_worker", "NATIVITY": "nativity"}

sha = hashlib.sha256(ZIP.read_bytes()).hexdigest()
with zipfile.ZipFile(ZIP) as z:
    name = next(n for n in z.namelist() if n.startswith("psam_p") and n.endswith(".csv"))
    with z.open(name) as f:
        df = pd.read_csv(f, usecols=["SERIALNO", "SPORDER", "ESR", "WKWN"] + VARS
                         + list(FLAGS.values()), dtype={"SERIALNO": str})

allocated = df[list(FLAGS.values())].eq(1).any(axis=1)
topcode = df.loc[df.WAGP.notna(), "WAGP"].max()
steps = [("household population (no GQ)", ~df.SERIALNO.str.contains("GQ")),
         ("age 25-64", df.AGEP.between(25, 64)),
         ("civilian employed (ESR 1,2)", df.ESR.isin([1, 2])),
         ("wage/salary worker (COW 1-5)", df.COW.isin([1, 2, 3, 4, 5])),
         ("WAGP > 0", df.WAGP > 0),
         ("full year (WKWN >= 50)", df.WKWN >= 50),
         ("no allocation on the 8 variables", ~allocated),
         (f"below top code (WAGP < {topcode:,.0f})", df.WAGP < topcode)]
mask = pd.Series(True, index=df.index)
counts = []
print(f"{name}: {len(df):,} person records; top-coded WAGP value = {topcode:,.0f}")
print("filters:")
for label, m in steps:
    mask &= m
    counts.append({"step": label, "remaining": int(mask.sum())})
    print(f"  after {label:<36} {mask.sum():>9,}")

pool = df[mask].sort_values(["SERIALNO", "SPORDER"]).reset_index(drop=True)
pool["educ"] = pd.NA
for lo, hi, lab in EDUC_BINS:
    pool.loc[pool.SCHL.between(lo, hi), "educ"] = lab
for v, mp in LABELS.items():
    pool[v] = pool[v].astype(int).map(mp)
pool = pool.rename(columns=RENAME)
cols = ["age", "hours", "wage", "sex", "marital", "educ", "class_worker", "nativity"]
pool = pool[cols].astype({"age": int, "hours": int, "wage": int})
assert pool.notna().all().all(), "unexpected missing values after recoding"

assert len(pool) >= N_ORG + N_TUNE, f"pool too small: {len(pool):,}"
perm = np.random.default_rng(SEED).permutation(len(pool))
d_org = pool.iloc[perm[:N_ORG]].reset_index(drop=True)
d_tune = pool.iloc[perm[N_ORG:N_ORG + N_TUNE]].reset_index(drop=True)


def summary(d):
    s = {"n": len(d)}
    for v in ["age", "hours", "wage"]:
        s[v] = {"mean": round(d[v].mean(), 2), "sd": round(d[v].std(), 2),
                "share_mode": round(d[v].eq(d[v].mode().iloc[0]).mean(), 3)}
    s["log_wage"] = {"mean": round(np.log(d.wage).mean(), 4), "skew": round(np.log(d.wage).skew(), 3)}
    for v in ["sex", "marital", "educ", "class_worker", "nativity"]:
        s[v] = d[v].value_counts(normalize=True).round(4).to_dict()
    return s


summ = {"pool": summary(pool), "d_org": summary(d_org), "d_tune": summary(d_tune)}
OUT.mkdir(parents=True, exist_ok=True)
d_org.to_csv(OUT / "d_org.csv", index=False)
d_tune.to_csv(OUT / "d_tune.csv", index=False)
manifest = {"source_url": URL, "source_file": name, "zip_sha256": sha, "seed": SEED,
            "n_org": N_ORG, "n_tune": N_TUNE, "pool_size": len(pool), "top_code": int(topcode),
            "adjinc_applied": False, "filters": counts,
            "educ_bins_schl": {lab: [lo, hi] for lo, hi, lab in EDUC_BINS},
            "labels": {RENAME[k]: v for k, v in LABELS.items()}, "summary": summ}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

print(f"\npool {len(pool):,} -> d_org {len(d_org):,}, d_tune {len(d_tune):,} (seed {SEED}); "
      f"written to {OUT}")
print("\n                       pool      d_org     d_tune")
for v in ["age", "hours", "wage"]:
    print(f"mean {v:<15} " + "".join(f"{summ[k][v]['mean']:>11,.1f}" for k in summ))
print(f"mean log wage        " + "".join(f"{summ[k]['log_wage']['mean']:>11.3f}" for k in summ))
print(f"share at 40 hours    " + "".join(f"{(summ[k]['hours']['share_mode']):>11.3f}" for k in summ))
for v in ["sex", "marital", "educ", "class_worker", "nativity"]:
    for lev in sorted(summ["pool"][v]):
        print(f"{v[:8]}={lev:<12}" + "".join(f"{summ[k][v].get(lev, 0):>11.3f}" for k in summ))
married = d_org.marital.eq("married")
print("\nd_org sex x married counts (E3 cells):")
print(pd.crosstab(d_org.sex, married.map({True: "married", False: "not_married"})).to_string())
print("d_org marital counts:", d_org.marital.value_counts().to_dict())
