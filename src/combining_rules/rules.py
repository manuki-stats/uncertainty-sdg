"""Combining rules for fully synthetic releases under Little's design.

From m synthetic files, each giving a point estimate q_i and its within-file
variance estimate u_i, we form

    q_bar = mean(q_i)          pooled point estimate
    b     = var(q_i, ddof=1)   between-file variance
    u_bar = mean(u_i)          mean within-file variance
    r     = n_syn / n_org      size ratio

and four variance estimators for q_bar:

    T_alt    = r*u_bar + b/m                  primary (Drechsler 2011)
    T_s      = (r + 1/m) * u_bar              plug-in, predicted (Raab et al. 2016)
    T_s_ppd  = (r + (1+r)/m) * u_bar          proper, predicted (Raab et al. 2016)
    T_f      = (1 + 1/m)*b - u_bar            background only (Reiter 2002)

T_p (Reiter 2003) is T_alt at r = 1 and needs no separate code.

target="population" gives intervals for the population parameter. For the real
extract, where the target is the complete-data estimate q_org, the sampling term
is dropped and only the synthesis variance remains ("sample" target).

The diagnostic kappa = b/u_bar is 1 under plug-in synthesis and 1 + r under
proper synthesis when the SDA holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

ESTIMATORS = ("T_alt", "T_s", "T_s_ppd", "T_f")
TARGETS = ("population", "sample")


@dataclass(frozen=True)
class Interval:
    estimator: str
    estimate: float
    variance: float
    df: float                 # np.inf means a normal quantile was used
    lo: float
    hi: float
    corrected: bool = False   # T_f only: the negative-variance correction fired

    def covers(self, target: float) -> bool:
        return bool(self.lo <= target <= self.hi)

    @property
    def width(self) -> float:
        return self.hi - self.lo


@dataclass(frozen=True)
class Pooled:
    q_bar: float
    b: float
    u_bar: float
    m: int
    r: float
    intervals: dict[str, Interval] = field(default_factory=dict)

    @property
    def kappa(self) -> float:
        return self.b / self.u_bar if self.u_bar > 0 else float("nan")


def _make_interval(name, est, var, df, alpha, corrected=False) -> Interval:
    if not (np.isfinite(est) and np.isfinite(var)) or var < 0:
        nan = float("nan")
        return Interval(name, est, nan, nan, nan, nan, corrected)
    if np.isfinite(df):
        crit = stats.t.ppf(1 - alpha / 2, df)
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
    half = crit * np.sqrt(var)
    return Interval(name, float(est), float(var), float(df),
                    float(est - half), float(est + half), corrected)


def naive(q: float, u: float, alpha: float = 0.05) -> Interval:
    """Step 1: one synthetic file analysed as if it were the original data."""
    return _make_interval("naive", q, u, np.inf, alpha)


def pool(q, u, *, r: float, alpha: float = 0.05, target: str = "population",
         estimators=ESTIMATORS) -> Pooled:
    """Pool m >= 2 estimates into one interval per requested estimator."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    q = np.asarray(q, dtype=float)
    u = np.asarray(u, dtype=float)
    m = q.size
    if m < 2 or u.size != m:
        raise ValueError(f"need m >= 2 matching estimates, got q={q.size}, u={u.size}")

    nan = float("nan")
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(u))):
        return Pooled(nan, nan, nan, m, r, {})

    q_bar = float(q.mean())
    b = float(q.var(ddof=1))
    u_bar = float(u.mean())
    out: dict[str, Interval] = {}

    for name in estimators:
        if target == "population":
            if name == "T_alt":
                var = r * u_bar + b / m
                df = np.inf if b == 0 else (m - 1) * (1 + r * u_bar / (b / m)) ** 2
                out[name] = _make_interval(name, q_bar, var, df, alpha)
            elif name == "T_s":
                out[name] = _make_interval(name, q_bar, (r + 1 / m) * u_bar, np.inf, alpha)
            elif name == "T_s_ppd":
                out[name] = _make_interval(name, q_bar, (r + (1 + r) / m) * u_bar, np.inf, alpha)
            elif name == "T_f":
                var = (1 + 1 / m) * b - u_bar
                if var > 0:
                    df = (m - 1) * (1 - u_bar / ((1 + 1 / m) * b)) ** 2
                    out[name] = _make_interval(name, q_bar, var, df, alpha)
                else:
                    out[name] = _make_interval(name, q_bar, r * u_bar, np.inf, alpha,
                                               corrected=True)
            else:
                raise ValueError(f"unknown estimator {name}")
        else:  # target == "sample": synthesis variance only
            if name == "T_alt":
                out[name] = _make_interval(name, q_bar, b / m, m - 1, alpha)
            elif name == "T_s":
                out[name] = _make_interval(name, q_bar, u_bar / m, np.inf, alpha)
            elif name == "T_s_ppd":
                out[name] = _make_interval(name, q_bar, (1 + r) * u_bar / m, np.inf, alpha)
            elif name == "T_f":
                continue  # not defined against the complete-data estimate
            else:
                raise ValueError(f"unknown estimator {name}")

    return Pooled(q_bar, b, u_bar, m, r, out)
