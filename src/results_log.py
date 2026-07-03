"""Append-only results log: one flat CSV, one row per (config, seed, estimand, metric).

Columns: config_hash, seed, method, dataset, step, estimand, m, K, R,
         metric, value, timestamp.

Rules (see README): never hand-edit; runs are idempotent (skip a (config_hash,
seed) already present) and resumable. The whole thesis reads from this file.
"""
from __future__ import annotations
from pathlib import Path

RESULTS_PATH = Path("results/results.csv")
COLUMNS = [
    "config_hash", "seed", "method", "dataset", "step",
    "estimand", "m", "K", "R", "metric", "value", "timestamp",
]


def append_row(row: dict) -> None:
    """Append one result row to the CSV, writing the header if the file is new."""
    import csv, datetime
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {**row, "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    new_file = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})


def already_done(config_hash: str, seed: int) -> bool:
    """True if this (config_hash, seed) already has rows, so a rerun can skip it."""
    if not RESULTS_PATH.exists():
        return False
    import csv
    with RESULTS_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["config_hash"] == config_hash and str(r["seed"]) == str(seed):
                return True
    return False
