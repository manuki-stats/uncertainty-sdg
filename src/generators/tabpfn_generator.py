"""TabPFN as a synthesiser, via tabpfn_extensions' TabPFNUnsupervisedModel.

How it generates (read from tabpfn-extensions 0.6.2): starting from an all-NaN
table, columns are filled one at a time in column order, each record drawn from
TabPFN's predictive distribution for that column given the columns already
generated, with the stored training data as in-context examples. Continuous
columns are sampled from TabPFN's discretised ("bar") predictive distribution
at temperature t; categorical columns from the predicted class probabilities
(the temperature does not apply to them). The predictive distributions are
averaged over `n_permutations` orderings of the conditioning columns.

"Fitting" stores the data as context; no weights change. Two consequences:
  - The only fit-level randomness is TabPFN's own ensemble preprocessing,
    controlled by `random_state`, which we tie to the training seed. A new
    training seed changes the fit only through that, so Step 3 is defined but
    expected to add little between-file scatter.
  - Step 4 (bootstrap) changes the context itself.
Each record integrates over TabPFN's approximation to the posterior predictive,
so its extra uncertainty is expected to show up in the within-file variance
u_bar rather than in the between-file variance b_m.

Column order matters for a sequential synthesiser. We use synthpop's visit
order (x2, x1, y), so y is drawn last given both predictors and the two
sequential generators are comparable.

Seeds: `random_state` of both TabPFN estimators = training seed. Sampling uses
torch's global RNG and the permutations use Python's `random`, so every
generate() call seeds those globally with the generation seed.

Warm-up: the first generation in a PROCESS does one-time initialisation that
draws from torch's global RNG, so without it the first file differs from every
later file with the same seed (measured: max difference 15 on y). The first
fit() in a process therefore generates a throwaway 2-row file; later fits skip
it, since it costs a full context pass (~20 s at 500 context rows on CPU).

Built-model cache: TabPFN refits (and by default rebuilds the network) once per
column per permutation, nine times per file. TABPFN_MODEL_CACHE_SIZE=2 keeps the
built classifier and regressor, which cut the cost per file by about a third.
Rebuilding also draws from torch's RNG, so the cache setting changes which
file a seed produces. The wrapper therefore sets it itself rather than leaving
it to the environment.

Size limits: TabPFN refuses CPU inference on more than 1,000 context rows by
default (too slow); `ignore_pretraining_limits=True` lifts that and the
10,000-row pretraining limit. Use a GPU for n_org in the thousands.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.seeds import set_global_seeds

from .base import Generator

ORDER = ("x2", "x1", "y")
MODEL_CACHE_SIZE = "2"
_WARMED_UP = False                                # per process, see the docstring
DISCRETE = ("x2",)


class TabPFNGenerator(Generator):
    name = "tabpfn"

    def __init__(self, temperature: float = 1.0, n_permutations: int = 3,
                 device: str = "auto", order=ORDER, discrete_columns=DISCRETE,
                 ignore_pretraining_limits: bool = True, params: dict | None = None,
                 warmup: bool = True):
        self.temperature = float(temperature)
        self.n_permutations = int(n_permutations)
        self.device = device
        self.order = list(order)
        self.discrete_columns = list(discrete_columns)
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.params = dict(params or {})          # extra TabPFN estimator kwargs
        self.warmup = warmup
        self._model = None

    def fit(self, data: pd.DataFrame, *, train_seed: int):
        global _WARMED_UP
        os.environ["TABPFN_MODEL_CACHE_SIZE"] = MODEL_CACHE_SIZE
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
        _silence_progress_bars()

        self._columns = list(data.columns)
        missing = set(self.order) ^ set(self._columns)
        if missing:
            raise ValueError(f"order {self.order} does not match columns {self._columns}")
        for c in self.discrete_columns:
            # Classes are sampled as indices 0..K-1 and used as values directly.
            levels = np.unique(data[c])
            if not np.array_equal(levels, np.arange(len(levels))):
                raise ValueError(f"discrete column {c!r} must be coded 0..K-1, got {levels}")

        kw = dict(random_state=int(train_seed), device=self.device,
                  ignore_pretraining_limits=self.ignore_pretraining_limits, **self.params)
        self._model = TabPFNUnsupervisedModel(tabpfn_clf=TabPFNClassifier(**kw),
                                              tabpfn_reg=TabPFNRegressor(**kw))
        self._model.set_categorical_features(
            [self.order.index(c) for c in self.discrete_columns])
        self._model.fit(data[self.order].to_numpy(dtype=np.float32))
        if self.warmup and not _WARMED_UP:
            set_global_seeds(int(train_seed))
            self._model.generate_synthetic_data(
                n_samples=2, t=self.temperature, n_permutations=self.n_permutations)
            _WARMED_UP = True
        return self

    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("generate() before fit()")
        set_global_seeds(int(gen_seed))
        out = self._model.generate_synthetic_data(
            n_samples=int(n), t=self.temperature, n_permutations=self.n_permutations)
        arr = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
        df = pd.DataFrame(arr, columns=self.order)[self._columns]
        return df.astype(float)


def _silence_progress_bars() -> None:
    """generate_synthetic_data wraps its column loop in tqdm; in a job log that
    is one progress bar per column per file. Replace it with a pass-through."""
    try:
        import tabpfn_extensions.unsupervised.unsupervised as U
        U.tqdm = lambda it, *a, **k: it
    except Exception:
        pass
