"""CTGAN and TVAE from the standalone `ctgan` package (sdv-dev), the reference
implementation of Xu et al. (2019).

Seeds: `set_random_state(int)` seeds numpy and torch together, scoped to the
next fit() or sample() call, so the training seed and the generation seed are
controlled separately without touching global state. That covers the CPU random
streams only: on a GPU, the latent noise in training and the Gumbel-softmax draws
(discrete columns and mode indicators, also at sampling) come from the CUDA
generator, which ctgan never seeds. With `enable_gpu`, the wrapper therefore
seeds CUDA with the same seed before each fit() and sample(), and by default
switches torch to deterministic kernels (`deterministic=True`), without which
cuBLAS may still sum in a different order from run to run. On CPU both are no-ops.

Discrete columns at generation (`fix_condvec`, CTGAN only): during training
CTGAN picks the conditional vector from log-flattened category frequencies
("training-by-sampling", Xu et al. 2019), so rare categories are seen more
often. ctgan 0.12.1 then reuses those log-flattened probabilities in
`sample_original_condvec`, whose own docstring says generation uses the
original frequencies: every categorical marginal is pulled toward uniform (x2:
0.40 -> 0.49). With `fix_condvec=True` the wrapper leaves training untouched and,
after fit(), replaces those probabilities with the empirical category shares of
the training data, so generation does what the package documents.

Both models treat every column as continuous unless told otherwise, so the
wrapper declares the discrete columns and casts them back to float 0/1 on the
way out, which is what the estimand code expects.

No process pools: ctgan's DataTransformer transforms columns with
`Parallel(n_jobs=-1)` whenever the data have 500 rows or more, i.e. a loky
process pool inside every worker. On the H100 nodes those pool processes
intermittently die at start-up (a named semaphore from /dev/shm is gone by the
time the child opens it: FileNotFoundError in SemLock._rebuild), which kills the
whole fit. The wrapper therefore runs fit() and sample() under joblib's
sequential backend: the same column transforms, in-process, one after another.
This is the path joblib already takes when the job scripts cap
LOKY_MAX_CPU_COUNT at 1, and on TVAE it was verified bit-identical to the
pooled path (rep 50), so results do not depend on it.

Hyperparameters default to the package defaults (epochs=300, batch_size=500);
the tuning search overrides them through `params`.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Deterministic cuBLAS needs this set before the first GPU matmul of the process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .base import Generator

DISCRETE = ("x2",)


class _CtganBase(Generator):
    _cls = None  # set by subclasses

    def __init__(self, params: dict | None = None, discrete_columns=DISCRETE,
                 enable_gpu: bool = False, deterministic: bool = True,
                 fix_condvec: bool = False):
        self.params = dict(params or {})
        self.discrete_columns = list(discrete_columns)
        self.enable_gpu = enable_gpu
        self.deterministic = deterministic
        self.fix_condvec = fix_condvec
        self._model = None

    def _seed_gpu(self, seed: int) -> None:
        """Seed the CUDA stream that ctgan's own random state does not reach."""
        if not self.enable_gpu:
            return
        import torch
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            if self.deterministic:
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False

    def fit(self, data: pd.DataFrame, *, train_seed: int):
        from ctgan import CTGAN, TVAE  # noqa: F401  (imported lazily: torch is heavy)
        self._columns = list(data.columns)
        self._model = self._cls(enable_gpu=self.enable_gpu, **self.params)
        self._model.set_random_state(int(train_seed))
        self._seed_gpu(train_seed)
        with _no_process_pools():
            self._model.fit(data, discrete_columns=self.discrete_columns)
        if self.fix_condvec:
            sampler = getattr(self._model, "_data_sampler", None)
            if sampler is None:
                raise ValueError(f"fix_condvec: {self.name} has no conditional vector")
            use_empirical_condvec(sampler)
        return self

    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("generate() before fit()")
        self._model.set_random_state(int(gen_seed))
        self._seed_gpu(gen_seed)
        with _no_process_pools():
            out = self._model.sample(int(n))
        out = out[self._columns].copy()
        for c in self.discrete_columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        return out.astype(float)


@contextmanager
def _no_process_pools():
    """Run any joblib Parallel inside the block in-process (see the docstring)."""
    from joblib import parallel_config
    with parallel_config(backend="sequential"):
        yield


def use_empirical_condvec(sampler) -> None:
    """Set a fitted ctgan DataSampler's category probabilities to the empirical
    shares of the training data, for generation only (training is over).

    `_rid_by_cat_cols[i][j]` holds the training rows in category j of discrete
    column i, in the same column order as the rows of
    `_discrete_column_category_prob`, so the counts are exact.
    """
    probs = sampler._discrete_column_category_prob
    for i, rows_by_cat in enumerate(sampler._rid_by_cat_cols):
        counts = np.array([len(r) for r in rows_by_cat], dtype=float)
        probs[i, :] = 0.0
        probs[i, : len(counts)] = counts / counts.sum()


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
