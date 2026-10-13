"""Step 1 of the real-data extract: profile the ACS 2024 1-year PUMS person file.

Usage (repo root, env synth2):  python scripts/acs_profile.py [state]   # default: ca
Downloads one state's person file to ./acs_raw/ (git-ignore it) and prints a profile of
the eight extract variables inside the proposed universe. Writes nothing else.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

YEAR = 2024
ST = (sys.argv[1] if len(sys.argv) > 1 else "ca").lower()
URL = f"https://www2.census.gov/programs-surveys/acs/data/pums/{YEAR}/1-Year/csv_p{ST}.zip"
RAW = Path("acs_raw")
ZIP = RAW / f"acs{YEAR}_1y_csv_p{ST}.zip"

NUM = ["AGEP", "WKHP", "WAGP"]                   # Adult: age, hours-per-week, income
CAT = ["SEX", "MAR", "SCHL", "COW", "NATIVITY"]  # Adult: sex, marital, education, workclass, country
EXTRACT = NUM + CAT
FLAG_OVERRIDE = {"NATIVITY": "FPOBP"}            # nativity is derived from place of birth
NEEDED = 10_000                                  # D_org (5000) + disjoint tuning sample (5000)

RAW.mkdir(exist_ok=True)
if not ZIP.exists():
    print(f"downloading {URL}")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(ZIP, "wb") as out:
        shutil.copyfileobj(r, out)

with zipfile.ZipFile(ZIP) as z:
    name = next(n for n in z.namelist() if n.startswith("psam_p") and n.endswith(".csv"))
    with z.open(name) as f:
        header = set(pd.read_csv(f, nrows=0).columns)
    flags = {}
    for v in EXTRACT:
        for cand in (FLAG_OVERRIDE.get(v), f"F{v}P", f"F{v}"):
            if cand in header:
                flags[v] = cand
                break
    wanted = ["SERIALNO", "ESR", "ADJINC", "WKWN"] + EXTRACT + list(flags.values())
    with z.open(name) as f:
        df = pd.read_csv(f, usecols=[c for c in wanted if c in header], dtype={"SERIALNO": str})

print(f"{name}: {len(df):,} person records")
print("requested but not in file:", sorted(set(wanted) - header) or "none")
print("allocation flags found:", flags, "| no flag for:", [v for v in EXTRACT if v not in flags] or "none")

# --- universe: civilian wage-and-salary workers aged 25-64 with positive wages, household pop. ---
steps = [("household population (no GQ)", ~df.SERIALNO.str.contains("GQ")),
         ("age 25-64", df.AGEP.between(25, 64)),
         ("civilian employed (ESR 1,2)", df.ESR.isin([1, 2])),
         ("wage/salary worker (COW 1-5)", df.COW.isin([1, 2, 3, 4, 5])),
         ("WAGP > 0", df.WAGP > 0)]
mask = pd.Series(True, index=df.index)
print("\nuniverse filter:")
for label, m in steps:
    mask &= m
    print(f"  after {label:<30} {mask.sum():>9,}")
u = df[mask]

# --- hidden missingness: values allocated (imputed) by the Census Bureau ---
alloc = pd.DataFrame({v: u[f].eq(1) for v, f in flags.items()})
print("\nallocation rate within universe:")
print(alloc.mean().sort_values(ascending=False).round(3).to_string())
clean = ~alloc.any(axis=1)
print(f"clean pool (no allocation on any extract variable): {clean.sum():,} ({clean.mean():.1%}); "
      f"needed >= {NEEDED:,}")
print(f"median WAGP, clean vs allocated: {u.WAGP[clean].median():,.0f} vs {u.WAGP[~clean].median():,.0f}")
print(f"median AGEP, clean vs allocated: {u.AGEP[clean].median():.0f} vs {u.AGEP[~clean].median():.0f}")
print("ADJINC values:", sorted(u.ADJINC.unique())[:5])

c = u[clean]
print("\nnumeric variables (clean pool):")
for v in NUM:
    x = c[v]
    pct = x.quantile([.01, .05, .25, .5, .75, .95, .99]).round(1).tolist()
    mode = x.mode().iloc[0]
    print(f"{v}: missing={x.isna().sum()}, min={x.min():g}, max={x.max():g}, mean={x.mean():.1f}, "
          f"sd={x.std():.1f}, skew={x.skew():.2f}, distinct={x.nunique()}")
    print(f"   p1..p99={pct}; mode={mode:g} ({x.eq(mode).mean():.1%}); "
          f"share at max={x.eq(x.max()).mean():.2%}")
lw = np.log(c.WAGP)
print(f"log WAGP: skew={lw.skew():.2f}, excess kurtosis={lw.kurt():.2f}; WAGP on multiples of "
      f"1000: {c.WAGP.mod(1000).eq(0).mean():.1%}, of 5000: {c.WAGP.mod(5000).eq(0).mean():.1%}")

print("\ncategorical variables (clean pool, shares by code):")
for v in CAT:
    s = c[v].value_counts(normalize=True, dropna=False).sort_index().round(3)
    print(f"{v}: {s.to_dict()}")
print("\nSEX x MAR counts (clean pool; E3 cells):")
print(pd.crosstab(c.SEX, c.MAR).to_string())

# --- optional full-year restriction: WKWN used only as a filter, not as a variable ---
if "WKWN" in c:
    fy = c.WKWN.ge(50)
    lw_fy = np.log(c.WAGP[fy])
    print(f"\nfull-year (WKWN >= 50): {fy.sum():,} ({fy.mean():.1%} of clean pool); "
          f"log WAGP skew={lw_fy.skew():.2f}, excess kurtosis={lw_fy.kurt():.2f}")
