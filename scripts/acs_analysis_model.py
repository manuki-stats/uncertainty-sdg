"""Step 3 (national): analysis-model diagnostics on the pseudo-population, plus a
design-based check of the complete-data inference at n_org.

Usage (repo root, env synth2):  python scripts/acs_analysis_model.py [n_draws]   # default 2000
Part 1 fits M1-M3 on the population (Q, classical vs HC3 SEs, residual checks).
Part 2 draws n_draws SRS of size n_org from the population and checks the complete-data
95% intervals for E1-E3 under M2 (variance ratio and coverage of Q): the real-data
counterpart of the oracle's complete-data check. Writes population_truth.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("extracts") / "acs2024_us"
N_ORG, SEED = 5000, 20261014
N_DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000

pop = pd.read_csv(OUT / "population.csv.gz")
EDUC_LEVELS = ["lt_hs", "some_college", "bachelor", "graduate"]  # reference: hs


def design(d, spec):
    a = ((d.age - 45) / 10).to_numpy(float)          # fixed constants, not data-dependent
    f = d.sex.eq("female").to_numpy(float)
    cols = {"const": np.ones(len(d)), "age": a, "female": f, "age:female": a * f}
    if spec in ("M2", "M3"):
        cols["age2"] = a ** 2
        cols.update({f"educ_{e}": d.educ.eq(e).to_numpy(float) for e in EDUC_LEVELS})
    if spec == "M3":
        cols["age2:female"] = a ** 2 * f
    return np.column_stack(list(cols.values())), list(cols)


def ols(X, y, hc3=True):
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    V_cl = (e @ e / (n - k)) * XtXi
    if not hc3:
        return b, e, V_cl, None
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    V_hc3 = XtXi @ ((X * (e ** 2 / (1 - h) ** 2)[:, None]).T @ X) @ XtXi
    return b, e, V_cl, V_hc3


y_pop = np.log(pop.wage.to_numpy(float))
print(f"population: N={len(pop):,}; mean log wage={y_pop.mean():.4f} (sd {y_pop.std(ddof=1):.3f})")

# ---- Part 1: specifications on the population ----
for spec in ("M1", "M2", "M3"):
    X, names = design(pop, spec)
    b, e, V_cl, V_hc3 = ols(X, y_pop)
    se_cl, se_hc = np.sqrt(np.diag(V_cl)), np.sqrt(np.diag(V_hc3))
    r2 = 1 - e @ e / ((y_pop - y_pop.mean()) @ (y_pop - y_pop.mean()))
    print(f"\n=== {spec}: k={len(names)}, R2={r2:.3f}, resid sd={e.std(ddof=len(names)):.3f}, "
          f"resid skew={stats.skew(e):.2f}, excess kurt={stats.kurtosis(e):.2f}")
    print(f"{'term':<18}{'coef':>9}{'hc3/cl':>8}")
    for nm, bi, sc, sh in zip(names, b, se_cl, se_hc):
        print(f"{nm:<18}{bi:>9.4f}{sh/sc:>8.2f}")
    band = pd.cut(pop.age, [24, 34, 44, 54, 64])
    r = pd.Series(e)
    print("resid mean by age band x sex:")
    print(r.groupby([band, pop.sex], observed=True).mean().unstack().round(3).to_string())
    print("resid sd by sex:", r.groupby(pop.sex).std().round(3).to_dict(),
          "| by educ:", r.groupby(pop.educ).std().round(3).to_dict())
    if spec == "M2":
        Q = {"E1": float(y_pop.mean()), "E2": float(b[names.index("age")]),
             "E3": float(b[names.index("age:female")])}

# ---- Part 2: design-based check of complete-data inference under M2 at n_org ----
rng = np.random.default_rng(SEED)
X_pop, names = design(pop, "M2")
i2, i3 = names.index("age"), names.index("age:female")
q = np.empty((N_DRAWS, 3))
u_cl = np.empty((N_DRAWS, 3))
u_hc = np.empty((N_DRAWS, 3))
for d in range(N_DRAWS):
    idx = rng.choice(len(pop), N_ORG, replace=False)
    X, y = X_pop[idx], y_pop[idx]
    b, _, V_cl, V_hc3 = ols(X, y)
    var_mean = y.var(ddof=1) / N_ORG
    q[d] = [y.mean(), b[i2], b[i3]]
    u_cl[d] = [var_mean, V_cl[i2, i2], V_cl[i3, i3]]
    u_hc[d] = [var_mean, V_hc3[i2, i2], V_hc3[i3, i3]]

f = N_ORG / len(pop)
Qv = np.array([Q["E1"], Q["E2"], Q["E3"]])
print(f"\ndesign-based check: {N_DRAWS} SRS of n={N_ORG} from N={len(pop):,} "
      f"(f = {f:.2%}; a valid variance overstates by about {1/(1-f):.3f})")
print(f"{'est':<4}{'Q':>10}{'bias/sd':>9}{'vr_cl':>8}{'vr_hc3':>8}{'cov_cl':>8}{'cov_hc3':>9}")
for j, est in enumerate(["E1", "E2", "E3"]):
    emp = q[:, j].var(ddof=1)
    cov = lambda u: np.mean(np.abs(q[:, j] - Qv[j]) <= 1.96 * np.sqrt(u[:, j]))
    print(f"{est:<4}{Qv[j]:>10.4f}{(q[:, j].mean() - Qv[j]) / np.sqrt(emp):>9.3f}"
          f"{u_cl[:, j].mean() / emp:>8.3f}{u_hc[:, j].mean() / emp:>8.3f}"
          f"{cov(u_cl):>8.3f}{cov(u_hc):>9.3f}")
print(f"(MC sd of a coverage near 0.95 at {N_DRAWS} draws: "
      f"{np.sqrt(0.95 * 0.05 / N_DRAWS):.3f}; of a variance ratio: {np.sqrt(2 / N_DRAWS):.3f})")

truth = {"spec": "M2: log wage ~ a + female + a:female + a^2 + educ (ref hs), a = (age-45)/10",
         "Q": Q, "population_size": len(pop), "n_org": N_ORG, "sampling_fraction": f}
(OUT / "population_truth.json").write_text(json.dumps(truth, indent=2))
print(f"wrote {OUT / 'population_truth.json'}")
