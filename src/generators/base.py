"""The interface every generator implements, so the harness never special-cases.

Training and generation take separate seeds. That separation is what lets the
four steps differ in exactly one thing each: Step 2 varies only the generation
seed, Step 3 the training seed too, Step 4 the data as well.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Generator(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, data: pd.DataFrame, *, train_seed: int) -> "Generator":
        """Fit to the (possibly bootstrapped) confidential data."""

    @abstractmethod
    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        """Draw one synthetic file of n rows from the fitted model."""
