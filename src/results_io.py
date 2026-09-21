"""Raw results: one CSV per chunk of replications, never one shared file.

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
