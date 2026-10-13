"""Shared definitions and a streaming reader for the ACS 2024 1-year PUMS national person file.

Imported by build_acs_extract.py. Downloads csv_pus.zip once to ./acs_raw/ (git-ignore it).
"""
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

YEAR = 2024
URL = f"https://www2.census.gov/programs-surveys/acs/data/pums/{YEAR}/1-Year/csv_pus.zip"
ZIP = Path("acs_raw") / f"acs{YEAR}_1y_csv_pus.zip"

VARS = ["AGEP", "WKHP", "WAGP", "SEX", "MAR", "SCHL", "COW", "NATIVITY"]
FLAGS = {"AGEP": "FAGEP", "WKHP": "FWKHP", "WAGP": "FWAGP", "SEX": "FSEXP", "MAR": "FMARP",
         "SCHL": "FSCHLP", "COW": "FCOWP", "NATIVITY": "FPOBP"}  # nativity derives from POBP
BASE = ["SERIALNO", "SPORDER", "ST", "ESR", "WKWN", "ADJINC"]  # "ST" resolved from the header
STATE_CANDIDATES = ["ST", "STATE"]

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
COLS = ["age", "hours", "wage", "sex", "marital", "educ", "class_worker", "nativity"]

STAGES = ["household population (no GQ)", "age 25-64", "civilian employed (ESR 1,2)",
          "wage/salary worker (COW 1-5)", "WAGP > 0"]


def download():
    ZIP.parent.mkdir(exist_ok=True)
    if not ZIP.exists():
        print(f"downloading {URL} (large; one-off)")
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        tmp = ZIP.with_suffix(".part")
        with urllib.request.urlopen(req) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out, length=1 << 22)
        tmp.rename(ZIP)


def _stage_masks(ch):
    return [~ch.SERIALNO.str.contains("GQ"), ch.AGEP.between(25, 64), ch.ESR.isin([1, 2]),
            ch.COW.isin([1, 2, 3, 4, 5]), ch.WAGP > 0]


def read_universe(chunksize=500_000):
    """Stream all person files; return (file names, base-universe rows, stage counts,
    per-state maximum WAGP over all persons = the state's top-code value)."""
    download()
    counts = {"all persons": 0, **dict.fromkeys(STAGES, 0)}
    keep, state_max = [], {}
    with zipfile.ZipFile(ZIP) as z:
        names = sorted(n for n in z.namelist() if n.startswith("psam_pus") and n.endswith(".csv"))
        with z.open(names[0]) as f:
            header = list(pd.read_csv(f, nrows=0).columns)
        st_col = next((c for c in STATE_CANDIDATES if c in header), None)
        if st_col is None:
            raise SystemExit("no state column found; header columns mentioning ST: "
                             f"{[c for c in header if 'ST' in c.upper()]}")
        needed = [st_col if c == "ST" else c for c in BASE] + VARS + list(FLAGS.values())
        missing = [c for c in needed if c not in header]
        if missing:
            raise SystemExit(f"columns not in {names[0]}: {missing}")
        print(f"state column: {st_col}")
        for name in names:
            with z.open(name) as f:
                reader = pd.read_csv(f, usecols=needed, dtype={"SERIALNO": str},
                                     chunksize=chunksize)
                for ch in reader:
                    ch = ch.rename(columns={st_col: "ST"})
                    for st, v in ch.groupby("ST").WAGP.max().items():
                        if pd.notna(v):
                            state_max[st] = max(state_max.get(st, v), v)
                    counts["all persons"] += len(ch)
                    m = pd.Series(True, index=ch.index)
                    for stage, cond in zip(STAGES, _stage_masks(ch)):
                        m &= cond
                        counts[stage] += int(m.sum())
                    keep.append(ch[m])
            print(f"  read {name}")
    return names, pd.concat(keep, ignore_index=True), counts, pd.Series(state_max)


def recode(df):
    """Map PUMS codes to the extract's columns and labels."""
    out = df.copy()
    out["educ"] = pd.NA
    for lo, hi, lab in EDUC_BINS:
        out.loc[out.SCHL.between(lo, hi), "educ"] = lab
    for v, mp in LABELS.items():
        out[v] = out[v].astype(int).map(mp)
    out = out.rename(columns=RENAME)[COLS]
    out = out.astype({"age": int, "hours": int, "wage": int})
    assert out.notna().all().all(), "unexpected missing values after recoding"
    return out
