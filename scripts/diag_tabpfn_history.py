"""One-off check (not part of the pipeline): a TabPFN file must not depend on
what ran earlier in the same process, or resuming a chunk in a fresh process
would reproduce different files.

Runs two fresh processes. A: fit (seed 11), generate (seed 1). B: first an
unrelated fit and file (seeds 99 and 5), then the same fit and file as A.
The two files must be identical.

    python scripts/diag_tabpfn_history.py
"""
import hashlib
import os
import subprocess
import sys
import time

CHILD = r'''
import sys, numpy as np, hashlib
sys.path.insert(0, ".")
from src.data.simulation import SimulationDGP
from src.generators.tabpfn_generator import TabPFNGenerator
d = SimulationDGP().sample(500, np.random.default_rng(0))
if sys.argv[1] == "B":
    TabPFNGenerator().fit(d, train_seed=99).generate(200, gen_seed=5)
f = TabPFNGenerator().fit(d, train_seed=11).generate(200, gen_seed=1)
print("HASH", hashlib.sha1(f.to_numpy().tobytes()).hexdigest())
'''

hashes = {}
for mode in ("A", "B"):
    t = time.perf_counter()
    out = subprocess.run([sys.executable, "-c", CHILD, mode], capture_output=True,
                         text=True, cwd=os.getcwd())
    line = [l for l in out.stdout.splitlines() if l.startswith("HASH")]
    if not line:
        print(f"process {mode} failed:\n{out.stderr[-2000:]}")
        sys.exit(1)
    hashes[mode] = line[0].split()[1]
    print(f"process {mode}: {hashes[mode][:12]}  ({time.perf_counter() - t:.0f}s)", flush=True)
print("history-independent:", hashes["A"] == hashes["B"])
