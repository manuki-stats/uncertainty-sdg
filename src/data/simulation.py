"""The simulation DGP: the only place in the thesis with a known ground truth.

    x1 ~ N(0, 1)
    x2 ~ Bernoulli(p)                      (kept as float 0/1, see note below)
    y  = b0 + b1*x1 + b2*x2 + b3*x1*x2 + e,   e ~ N(0, sigma^2)

Because x1 and x2 are independent and E[x1] = 0, the marginal mean of y is
b0 + b2*p in closed form, so all three estimands have analytic targets and
coverage is verifiable rather than approximated.

x2 stays float rather than categorical on purpose: patsy names the interaction
term "x1:x2" for a numeric x2 and "x1:x2[T.1]" for a categorical one, and the
estimand lookup depends on that name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FORMULA = "y ~ x1 * x2"
FOCAL = "y"
MAIN_TERM = "x1"
HARD_TERM = "x1:x2"


@dataclass(frozen=True)
class SimulationDGP:
    b0: float = 1.0
    b1: float = 2.0
    b2: float = -0.5
    b3: float = 1.5
    p: float = 0.4
    sigma: float = 1.0

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw n rows. The rng is passed in, never global, so runs are exactly
        reproducible from the seed logged with each result row."""
        x1 = rng.normal(size=n)
        x2 = rng.binomial(1, self.p, size=n).astype(float)
        y = (
            self.b0
            + self.b1 * x1
            + self.b2 * x2
            + self.b3 * x1 * x2
            + rng.normal(scale=self.sigma, size=n)
        )
        return pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    def truth(self) -> dict[str, float]:
        """Population values of E1-E3, the targets coverage is measured against."""
        return {"E1": self.b0 + self.b2 * self.p, "E2": self.b1, "E3": self.b3}


if __name__ == "__main__":
    dgp = SimulationDGP()
    rng = np.random.default_rng(0)
    df = dgp.sample(200_000, rng)

    from src.evaluation.estimands import e1_mean, e2_coefficient, e3_hard

    got = {
        "E1": e1_mean(df, FOCAL)[0],
        "E2": e2_coefficient(df, FORMULA, MAIN_TERM)[0],
        "E3": e3_hard(df, FORMULA, HARD_TERM)[0],
    }
    for k, target in dgp.truth().items():
        print(f"{k}: truth = {target: .4f}   large-sample estimate = {got[k]: .4f}")