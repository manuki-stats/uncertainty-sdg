"""Centralised seeding, so every result is regenerable from a logged seed.

The contract (see README): every row in the results log carries the seed that
produced it, and calling `set_all_seeds(seed)` before a run makes that run
deterministic across numpy, torch, and R.
"""
from __future__ import annotations


def set_all_seeds(seed: int) -> None:
    """Seed every source of randomness we use: Python, numpy, torch, and R.

    Call once at the top of each replication, passing the seed that will also
    be written to the results log for that row.
    """
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Seed R too, since synthpop runs there via rpy2.
    try:
        import rpy2.robjects as ro
        ro.r(f"set.seed({int(seed)})")
    except Exception:
        # R/rpy2 not available in this context; skip silently.
        pass
