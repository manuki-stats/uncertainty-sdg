"""The real-data pseudo-population: ACS 2024 1-year PUMS, national.

Built by scripts/build_acs_extract.py (see extracts/acs2024_us/manifest.json):
civilian wage-and-salary workers aged 25-64 in households, full year, all eight
items reported, wages below the state top code. N = 717,219 after a disjoint
tuning sample of 5,000 was reserved.

Design (a design-based simulation from a pseudo-population): every replication
draws its own D_org as a simple random sample without replacement from the
population (f = n_org / N = 0.70%), so the population's M2 coefficients are the
targets Q, exactly as the DGP's parameters are in the simulation. The q_org
target is available from the same runs (see run.py's orig_* files).

Coding. Generators see eight numeric columns in ORDER. Discrete columns are
coded 0..K-1 as floats, the convention the simulation used for x2 and the one
TabPFN's wrapper requires; LEVELS gives the labels. log_wage = log(wage) is
synthesised directly (no positivity constraint for any generator).

ORDER is also the visit order of the two sequential synthesisers (synthpop,
TabPFN): demographics, education, marital status, job, hours, and the outcome
last, as y is last in the simulation.

The analysis model M2, with a = (age - 45) / 10 (fixed constants, so the
transformation is the same for every file):

    log_wage ~ a + female + a:female + a^2 + educ dummies (reference: hs)

    E1 mean of log_wage   E2 coefficient of a   E3 coefficient of a:female
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACT = Path("extracts/acs2024_us")

LEVELS = {
    "sex": ("male", "female"),
    "nativity": ("native", "foreign_born"),
    "educ": ("lt_hs", "hs", "some_college", "bachelor", "graduate"),
    "marital": ("married", "widowed", "divorced", "separated", "never_married"),
    "class_worker": ("private_profit", "private_nonprofit", "local_gov", "state_gov",
                     "federal_gov"),
}
ORDER = ("sex", "age", "nativity", "educ", "marital", "class_worker", "hours", "log_wage")
DISCRETE = ("sex", "nativity", "educ", "marital", "class_worker")
CONTINUOUS = ("age", "hours", "log_wage")

FEMALE = float(LEVELS["sex"].index("female"))
EDUC_REF = "hs"
AGE_CENTRE, AGE_SCALE = 45.0, 10.0
EDUC_DUMMIES = tuple(f"e_{lab}" for lab in LEVELS["educ"] if lab != EDUC_REF)

# M2 written on the derived frame; defines the pMSE regressors for tuning.
FEATURE_FORMULA = "log_wage ~ a * female + a2 + " + " + ".join(EDUC_DUMMIES)


def code(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract columns (labels, wage in dollars) -> the generators' numeric frame."""
    out = {}
    for c in ORDER:
        if c == "log_wage":
            out[c] = np.log(raw["wage"].to_numpy(dtype=float))
        elif c in DISCRETE:
            mapping = {lab: float(i) for i, lab in enumerate(LEVELS[c])}
            col = raw[c].map(mapping)
            if col.isna().any():
                bad = sorted(set(raw[c][col.isna()]))
                raise ValueError(f"unknown level(s) in {c!r}: {bad}")
            out[c] = col.to_numpy(dtype=float)
        else:
            out[c] = raw[c].to_numpy(dtype=float)
    return pd.DataFrame(out)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """The analysis model's regressors, from a coded (real or synthetic) frame."""
    a = (df["age"].to_numpy(dtype=float) - AGE_CENTRE) / AGE_SCALE
    d = {"log_wage": df["log_wage"].to_numpy(dtype=float), "a": a,
         "female": (df["sex"].to_numpy(dtype=float) == FEMALE).astype(float), "a2": a ** 2}
    educ = df["educ"].to_numpy(dtype=float)
    for i, lab in enumerate(LEVELS["educ"]):
        if lab != EDUC_REF:
            d[f"e_{lab}"] = (educ == i).astype(float)
    return pd.DataFrame(d)


def estimands_acs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """E1-E3 under M2 with classical OLS variances; (q, u) ordered E1, E2, E3.

    The fast path for the real data, pinned against statsmodels in
    tests/test_acs.py. An education level a file never produces has an all-zero
    dummy; it is dropped, which is what patsy does for an unobserved level, so
    E2 and E3 stay defined. If the design is still rank-deficient (no women, a
    constant age), E2 and E3 are NaN: a logged result, not a crash.
    """
    d = derive(df)
    y = d["log_wage"].to_numpy()
    n = y.size
    nan = float("nan")
    if n < 2 or not np.all(np.isfinite(y)):
        return np.full(3, nan), np.full(3, nan)
    e1 = (float(y.mean()), float(y.var(ddof=1) / n))

    a, f = d["a"].to_numpy(), d["female"].to_numpy()
    cols = [np.ones(n), a, f, a * f, d["a2"].to_numpy()]
    cols += [d[c].to_numpy() for c in EDUC_DUMMIES if d[c].to_numpy().any()]
    X = np.column_stack(cols)
    if not np.all(np.isfinite(X)) or np.linalg.matrix_rank(X) < X.shape[1] or n <= X.shape[1]:
        return np.array([e1[0], nan, nan]), np.array([e1[1], nan, nan])
    G = np.linalg.inv(X.T @ X)
    beta = G @ (X.T @ y)
    resid = y - X @ beta
    s2 = float(resid @ resid) / (n - X.shape[1])
    var_beta = s2 * np.diag(G)
    return (np.array([e1[0], beta[1], beta[3]]),
            np.array([e1[1], var_beta[1], var_beta[3]]))


@lru_cache(maxsize=2)
def load_population(path: str = str(EXTRACT / "population.csv.gz")) -> pd.DataFrame:
    """Coded population, read once per process (run.py asks for it every replication)."""
    return code(pd.read_csv(path))


def load_tuning_sample(path: str = str(EXTRACT / "d_tune.csv")) -> pd.DataFrame:
    return code(pd.read_csv(path))


@lru_cache(maxsize=2)
def _truth(path: str) -> tuple[float, float, float]:
    q, _ = estimands_acs(load_population(path))
    stored_path = Path(path).with_name("population_truth.json")
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())["Q"]
        for name, v in zip(("E1", "E2", "E3"), q):
            if abs(v - stored[name]) > 1e-9:
                raise RuntimeError(f"{name}: population gives {v!r}, {stored_path} says "
                                   f"{stored[name]!r}; extract and truth out of sync")
    return tuple(float(v) for v in q)


class ACSPopulation:
    """Pseudo-population with the same interface as SimulationDGP."""

    def __init__(self, path: str | Path = EXTRACT / "population.csv.gz"):
        self.path = str(path)
        self.data = load_population(self.path)

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """n rows, simple random sampling without replacement, in draw order."""
        idx = rng.choice(len(self.data), size=n, replace=False)
        return self.data.iloc[idx].reset_index(drop=True)

    def truth(self) -> dict[str, float]:
        """Population values of E1-E3 under M2 (checked against population_truth.json)."""
        return dict(zip(("E1", "E2", "E3"), _truth(self.path)))
