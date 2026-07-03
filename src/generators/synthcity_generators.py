"""Wrap synthcity plugins (CTGAN, TVAE, TabDDPM=ddpm) behind the Generator interface.

Note the plugin name for diffusion is "ddpm", not "tabddpm".
Implement fit/generate when we build the harness.
"""
from __future__ import annotations
import pandas as pd
from .base import Generator

PLUGIN_NAMES = {"ctgan": "ctgan", "tvae": "tvae", "ddpm": "ddpm"}


class SynthcityGenerator(Generator):
    """Thin wrapper around synthcity's Plugins().get(name)."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._kwargs = kwargs
        self._model = None

    def fit(self, data: pd.DataFrame) -> "SynthcityGenerator":
        raise NotImplementedError("Wrap Plugins().get(name).fit(loader).")

    def generate(self, n: int) -> pd.DataFrame:
        raise NotImplementedError("Wrap model.generate(count=n).dataframe().")
