"""Seeding, so every result is regenerable from (seed0, replication).

Every random quantity in a replication gets its own seed, derived from
(seed0, rep, step, purpose, idx) through numpy's SeedSequence. The streams are
statistically independent, and a seed depends only on its own coordinates, so:

  - the confidential sample of replication `rep` is the same in every step
    (step 0 is reserved for it), which is what makes the steps comparable;
  - training and generation seeds are controlled separately, so Step 2 and
    Step 3 differ in exactly one thing;
  - adding a replication or a step never changes the seeds of existing ones.

`set_global_seeds` is kept for libraries that only read global state
(torch, R). The oracle never needs it.
"""
from __future__ import annotations

import numpy as np

PURPOSES = {"data": 0, "train": 1, "gen": 2, "boot": 3}


def derive_seed(seed0: int, rep: int, step: int, purpose: str, idx: int = 0) -> int:
    """A 32-bit seed unique to (seed0, rep, step, purpose, idx)."""
    ss = np.random.SeedSequence(entropy=seed0,
                                spawn_key=(rep, step, PURPOSES[purpose], idx))
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def set_global_seeds(seed: int) -> None:
    """Seed the global state of Python, numpy, torch and R, where present."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import rpy2.robjects as ro
        ro.r(f"set.seed({int(seed)})")
    except Exception:
        pass
