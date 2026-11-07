"""Which chunks of a cell are done, without asking SLURM.

  python -m src.harness.progress CONFIG START:STOP [--chunk-size 10]

Reads the chunk files in results/raw/<name>__<hash>/ and reports how many of
the expected chunks exist, which ranges are missing, and which code versions
produced the rows (more than one version in a cell means mixed code).
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.config import ExperimentConfig
from src.results_io import cell_dir, chunk_done, chunk_path


def _compress(ranges: list[tuple[int, int]]) -> str:
    """Merge adjacent missing chunks into start:stop spans."""
    spans: list[list[int]] = []
    for a, b in ranges:
        if spans and spans[-1][1] == a:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    return ", ".join(f"{a}:{b}" for a, b in spans)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("reps", help="start:stop, stop exclusive")
    ap.add_argument("--chunk-size", type=int, default=10)
    a = ap.parse_args()
    cfg = ExperimentConfig.from_yaml(a.config)
    h = cfg.hash()
    start, stop = (int(x) for x in a.reps.split(":"))

    done, missing = [], []
    for c0 in range(start, stop, a.chunk_size):
        c1 = min(c0 + a.chunk_size, stop)
        (done if chunk_done(chunk_path(cfg.name, h, c0, c1)) else missing).append((c0, c1))
    print(f"{cfg.name}__{h}: {len(done)}/{len(done) + len(missing)} chunks done")
    if missing:
        print(f"  missing reps: {_compress(missing)}")

    versions = {}
    for c0, c1 in done:
        path = chunk_path(cfg.name, h, c0, c1)
        for v in pd.read_csv(path, usecols=["code_version"], dtype={"code_version": str})["code_version"].unique():
            versions[v] = versions.get(v, 0) + 1
    if versions:
        print("  code versions (chunks): " + ", ".join(f"{v} ({n})" for v, n in versions.items()))
        if len(versions) > 1:
            print("  -> more than one code version in this cell")
    extra = sorted(p.name for p in cell_dir(cfg.name, h).glob("reps_*.csv")
                   if p not in {chunk_path(cfg.name, h, c0, c1) for c0, c1 in done})
    if extra:
        print(f"  {len(extra)} other chunk files (another range or chunk size): {extra[:3]}...")


if __name__ == "__main__":
    main()
