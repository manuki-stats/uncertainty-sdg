"""Wrap native R synthpop (via rpy2) behind the Generator interface --- the control.

This is the trusted positive control: synthpop's proper (Bayesian-bootstrap) CART
synthesis should recover near-nominal coverage under multiple synthesis. The load-
bearing check (see checklist phase 4) is that we invoke PROPER synthesis, not
deterministic CART. Implement the pandas<->R hand-off when we build the harness.
"""
from __future__ import annotations
import pandas as pd
from .base import Generator


class SynthpopGenerator(Generator):
    name = "synthpop"

    def __init__(self, method: str = "cart", proper: bool = True, **kwargs):
        # proper=True selects the Bayesian-bootstrap proper-synthesis mode.
        self.method = method
        self.proper = proper
        self._kwargs = kwargs
        self._fitted = None

    def fit(self, data: pd.DataFrame) -> "SynthpopGenerator":
        raise NotImplementedError("Pass the frame to R; call synthpop::syn(..., proper=TRUE).")

    def generate(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Pull the synthetic frame back from R to pandas.")
