"""CTGAN and TVAE from the standalone `ctgan` package (sdv-dev), the reference
implementation of Xu et al. (2019).

Seeds: `set_random_state(int)` seeds numpy and torch together, scoped to the
next fit() or sample() call, so the training seed and the generation seed are
controlled separately without touching global state. On CPU this is
deterministic; on a GPU, torch may need deterministic algorithms enabled.

Both models treat every column as continuous unless told otherwise, so the
wrapper declares the discrete columns and casts them back to float 0/1 on the
way out, which is what the estimand code expects.

Hyperparameters default to the package defaults (epochs=300, batch_size=500);
the tuning search overrides them through `params`.
"""
from __future__ import annotations

import pandas as pd

from .base import Generator

DISCRETE = ("x2",)


class _CtganBase(Generator):
    _cls = None  # set by subclasses

    def __init__(self, params: dict | None = None, discrete_columns=DISCRETE,
                 enable_gpu: bool = False):
        self.params = dict(params or {})
        self.discrete_columns = list(discrete_columns)
        self.enable_gpu = enable_gpu
        self._model = None

    def fit(self, data: pd.DataFrame, *, train_seed: int):
        from ctgan import CTGAN, TVAE  # noqa: F401  (imported lazily: torch is heavy)
        self._columns = list(data.columns)
        self._model = self._cls(enable_gpu=self.enable_gpu, **self.params)
        self._model.set_random_state(int(train_seed))
        self._model.fit(data, discrete_columns=self.discrete_columns)
        return self

    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("generate() before fit()")
        self._model.set_random_state(int(gen_seed))
        out = self._model.sample(int(n))
        out = out[self._columns].copy()
        for c in self.discrete_columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out.astype(float)


class CTGANGenerator(_CtganBase):
    name = "ctgan"

    @property
    def _cls(self):
        from ctgan import CTGAN
        return CTGAN


class TVAEGenerator(_CtganBase):
    name = "tvae"

    @property
    def _cls(self):
        from ctgan import TVAE
        return TVAE
