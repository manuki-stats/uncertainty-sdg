"""The parametric oracle: a correctly-specified generator with known behaviour.

This is NOT a method under test. It is a generator we know is correct, used to
validate the harness: under the fully-synthetic combining rules it must yield
~95% coverage on E1-E3. If it does, the combining-rules code is trustworthy and
every later result stands on solid ground. If it does not, the bug is ours, in
the variance code --- and we find it before any neural net is involved.
"""
from __future__ import annotations
import pandas as pd
from .base import Generator


class ParametricOracle(Generator):
    name = "oracle"

    def fit(self, data: pd.DataFrame) -> "ParametricOracle":
        raise NotImplementedError("Fit the known parametric model to the data.")

    def generate(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Draw from the correctly-specified model.")
