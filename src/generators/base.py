"""A common interface for every generator, so the harness treats them uniformly.

The harness needs exactly two things from any generator: fit it to real data,
then draw a synthetic dataset. synthcity models (CTGAN/TVAE/ddpm) and R synthpop
(via rpy2) both wrap to this interface, so the evaluation loop never special-cases.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd


class Generator(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "Generator":
        ...

    @abstractmethod
    def generate(self, n: int) -> pd.DataFrame:
        """Draw one synthetic dataset of n rows from the fitted model."""
        ...
