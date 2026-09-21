"""The gate, as a fast test: R = 200, Steps 1-2, proper and plug-in oracles.

The full gate is the R = 1000 run through the harness; this catches regressions
in seconds. Bands are about three Monte Carlo standard errors wide.
"""
import numpy as np
import pytest

from src.config import ExperimentConfig
from src.harness.run import run_replication

R = 200


def _collect(mode):
    cfg = ExperimentConfig(name="gate", generator="oracle",
                           generator_kwargs={"mode": mode}, n_org=1000, r=2,
                           m=10, K=10, steps=(1, 2), estimators=("T_alt",))
    rows = [row for rep in range(R) for row in run_replication(cfg, rep)]
    return rows


@pytest.mark.parametrize("mode, kappa_expected", [("proper", 3.0), ("plugin", 1.0)])
def test_gate(mode, kappa_expected):
    rows = _collect(mode)
    for est in ("E1", "E2", "E3"):
        step2 = [r for r in rows if r["step"] == 2 and r["estimand"] == est]
        step1 = [r for r in rows if r["step"] == 1 and r["estimand"] == est]
        cov2 = np.mean([r["covered"] for r in step2])
        cov1 = np.mean([r["covered"] for r in step1])
        kappa = np.mean([r["kappa"] for r in step2])
        assert 0.90 <= cov2 <= 0.99, f"{mode} {est}: T_alt coverage {cov2:.3f}"
        assert cov1 < 0.85, f"{mode} {est}: naive coverage {cov1:.3f} should fail"
        assert kappa == pytest.approx(kappa_expected, rel=0.15)
