"""Environment check: package versions, torch/CUDA, and the R bridge.

Run from the repo root with the env active:
    python scripts/env_check.py            # login node: CPU checks + R bridge
    srun ... python scripts/env_check.py   # GPU node: also checks CUDA

The synthpop check goes through the harness's own SynthpopGenerator, so it tests
the exact code path the experiments use: rpy2 -> native R syn(). Exit code 1 if
any check fails (a missing GPU is reported, not failed).
"""
from __future__ import annotations

import importlib
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FAILED: list[str] = []


def check(name, fn):
    try:
        msg = fn()
        print(f"[ OK ] {name}: {msg}")
    except Exception as e:  # noqa: BLE001 - report everything, keep going
        FAILED.append(name)
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def versions():
    from importlib.metadata import version
    out = []
    for mod, dist in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                      ("statsmodels", "statsmodels"), ("patsy", "patsy"),
                      ("yaml", "pyyaml"), ("torch", "torch"), ("ctgan", "ctgan"),
                      ("tabpfn", "tabpfn"), ("tabpfn_extensions", "tabpfn-extensions"),
                      ("rpy2", "rpy2")]:
        importlib.import_module(mod)          # the import is the check
        out.append(f"{dist} {version(dist)}")
    return "; ".join(out)


def torch_info():
    import torch
    msg = f"threads={torch.get_num_threads()}, cuda build={torch.version.cuda}"
    if not torch.cuda.is_available():
        return msg + ", GPU: none visible (fine on CPU nodes)"
    x = torch.randn(2048, 2048, device="cuda")
    (x @ x).sum().item()                  # first call pays CUDA/cuBLAS start-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(10):
        (x @ x).sum().item()
    torch.cuda.synchronize()
    return (msg + f", GPU: {torch.cuda.get_device_name(0)}, "
            f"2048x2048 matmul {100 * (time.perf_counter() - t):.2f} ms each")


def r_info():
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    importr("synthpop")
    rv = ro.r("R.version.string")[0]
    sv = ro.r('as.character(packageVersion("synthpop"))')[0]
    return f"{rv}; synthpop {sv}; R_HOME={ro.r('R.home()')[0]}"


def synthpop_roundtrip():
    import numpy as np
    from src.data.simulation import SimulationDGP
    from src.generators.synthpop_generator import SynthpopGenerator

    d = SimulationDGP().sample(5000, np.random.default_rng(0))
    g = SynthpopGenerator(proper=True).fit(d, train_seed=1)
    t = time.perf_counter()
    a = g.generate(10000, gen_seed=2)
    secs = time.perf_counter() - t
    b = g.generate(10000, gen_seed=2)
    c = g.generate(10000, gen_seed=3)
    assert list(a.columns) == ["y", "x1", "x2"] and len(a) == 10000
    assert a.equals(b), "same seeds gave different files (R seeding broken)"
    assert not a.equals(c), "different generation seeds gave identical files"
    return f"n_org 5000 -> 10000 rows in {secs:.1f} s; seeding reproducible"


print(f"python {platform.python_version()} on {platform.node()}")
check("imports", versions)
check("torch", torch_info)
check("R + synthpop", r_info)
check("synthpop via harness", synthpop_roundtrip)
print("ALL OK" if not FAILED else f"FAILED: {', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
