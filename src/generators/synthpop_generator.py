"""synthpop-CART via rpy2, calling the native R package.

synthpop fits and generates in ONE call (`syn`), so fit() only stores the data
and generate() does the work. Consequences for the release designs:

  proper=TRUE   every file starts from its own bootstrap of the data, so each
                file is a fresh fit -> proper synthesis. Step 2 is then already
                proper; expect kappa ~ 1 + r and nominal coverage. The control.
  proper=FALSE  CART is deterministic given the data, so files differ only by
                resampling within leaves -> plug-in synthesis; expect kappa ~ 1.

The seed passed to syn() combines the training and generation seeds, so the
harness's seed bookkeeping still holds: Step 2 varies the generation seed only,
Step 3 both. Step 4's harness-level bootstrap plus proper=TRUE bootstraps
twice; use proper=FALSE in Step 4 if you want exactly one.

Columns are passed in `order` (default x2, x1, y, so that y is synthesised
last, conditional on both predictors; synthpop visits columns in order) and
returned in the caller's column order. `discrete_columns` (coded 0..K-1) go to
R as factors. The defaults reproduce the simulation exactly; the real data set
both from src/data/acs.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Generator

ORDER = ("x2", "x1", "y")
DISCRETE = ("x2",)


class SynthpopGenerator(Generator):
    name = "synthpop"

    def __init__(self, proper: bool = True, method: str = "cart",
                 params: dict | None = None, order=ORDER, discrete_columns=DISCRETE):
        self.proper = proper
        self.method = method
        self.params = dict(params or {})   # extra syn() args, e.g. {"cart.minbucket": 5}
        self.order = list(order)
        self.discrete_columns = list(discrete_columns)
        self._data = None

    def fit(self, data: pd.DataFrame, *, train_seed: int):
        self._columns = list(data.columns)
        if set(self.order) != set(self._columns):
            raise ValueError(f"order {self.order} does not match columns {self._columns}")
        self._data = data[self.order].astype(float).reset_index(drop=True)
        # Pass discrete columns to R as factors so CART treats them as categorical
        # and synthpop prints no conversion note. rpy2 converts a Categorical to an
        # R factor only when its categories are STRINGS, hence the astype(str).
        for c in self.discrete_columns:
            codes = self._data[c].astype(int)
            levels = [str(v) for v in sorted(codes.unique())]
            self._data[c] = pd.Categorical(codes.astype(str), categories=levels)
        self._train_seed = int(train_seed)
        return self

    def generate(self, n: int, *, gen_seed: int) -> pd.DataFrame:
        if self._data is None:
            raise RuntimeError("generate() before fit()")
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects.packages import importr

        synthpop = importr("synthpop")
        seed = int(np.random.SeedSequence([self._train_seed, int(gen_seed)])
                   .generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))

        # Convert with pandas2ri only at the edges. Calling syn() while the
        # pandas converter is active turns its result into a NamedList
        # (rpy2 >= 3.6), which has no rx2().
        pd_conv = ro.default_converter + pandas2ri.converter
        with localconverter(pd_conv):
            r_df = ro.conversion.get_conversion().py2rpy(self._data)
        res = synthpop.syn(r_df, method=self.method, m=1, k=int(n),
                           proper=self.proper, seed=seed,
                           **{"print.flag": False, **self.params})
        with localconverter(pd_conv):
            out = ro.conversion.get_conversion().rpy2py(res.rx2("syn"))

        out = pd.DataFrame(out)
        for c in self.order:   # factors come back as categoricals of strings
            if isinstance(out[c].dtype, pd.CategoricalDtype) or out[c].dtype == object:
                out[c] = pd.to_numeric(out[c].astype(str))
        out = out[self.order].astype(float).reset_index(drop=True)
        return out[self._columns]
