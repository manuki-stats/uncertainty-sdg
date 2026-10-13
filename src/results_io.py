"""Raw results: one CSV per chunk of replications, never one shared file.

Each chunk has a companion files_*.csv with the per-file estimates (q, u) of
every step, and an orig_*.csv with the estimates on each replication's
confidential sample (q_org, u_org), both written before the chunk itself, so an
existing chunk always has them. Chunks written before 13 Oct 2026 have no
orig_*.csv.

Each chunk is written atomically (temp file, then rename), so a killed job never
leaves a half-written file, and parallel SLURM tasks never write to the same
file. A chunk that exists is complete; rerunning skips it. Never hand-edit.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

RAW_ROOT = Path("results/raw")


def cell_dir(name: str, config_hash: str) -> Path:
    return RAW_ROOT / f"{name}__{config_hash}"


def chunk_path(name: str, config_hash: str, start: int, stop: int) -> Path:
    return cell_dir(name, config_hash) / f"reps_{start:05d}-{stop - 1:05d}.csv"


def files_path(name: str, config_hash: str, start: int, stop: int) -> Path:
    """Per-file estimates for the same range as chunk_path."""
    return cell_dir(name, config_hash) / f"files_{start:05d}-{stop - 1:05d}.csv"


def orig_path(name: str, config_hash: str, start: int, stop: int) -> Path:
    """Confidential-sample estimates for the same range as chunk_path."""
    return cell_dir(name, config_hash) / f"orig_{start:05d}-{stop - 1:05d}.csv"


def chunk_done(path: Path) -> bool:
    return path.exists()


def write_chunk(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_cell(name: str, config_hash: str) -> pd.DataFrame:
    files = sorted(cell_dir(name, config_hash).glob("reps_*.csv"))
    if not files:
        raise FileNotFoundError(f"no results for {name}__{config_hash}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def read_cell_files(name: str, config_hash: str) -> pd.DataFrame:
    """Per-file estimates of a cell: rep, step, file, estimand, q, u, n_rejected."""
    files = sorted(cell_dir(name, config_hash).glob("files_*.csv"))
    if not files:
        raise FileNotFoundError(f"no per-file results for {name}__{config_hash}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def read_cell_orig(name: str, config_hash: str) -> pd.DataFrame:
    """Confidential-sample estimates of a cell: rep, estimand, q, u."""
    files = sorted(cell_dir(name, config_hash).glob("orig_*.csv"))
    if not files:
        raise FileNotFoundError(f"no orig_* results for {name}__{config_hash}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
