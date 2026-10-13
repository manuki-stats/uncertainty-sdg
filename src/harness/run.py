"""The evaluation loop: the reusable core of the thesis.

One replication draws a confidential sample, then runs each requested step:

  Step 1  one fit, one file, analysed as if real               (naive interval)
  Step 2  the same fit, m files, generation seed varies        (pooled over m)
  Step 3  K fits on the same data, training seed varies        (pooled over K)
  Step 4  K fits on K bootstrap resamples of the data          (pooled over K)

Steps 1 and 2 share one fit, and Step 1 analyses the first of Step 2's files.
Every step of a replication sees the same confidential sample.

Datasets: "simulation" (the DGP, analytic truth) and "acs" (the ACS
pseudo-population; each replication draws its own D_org without replacement,
truth = the population's M2 coefficients; see src/data/acs.py).

Besides the result chunk and the per-file table, every chunk writes orig_*.csv:
the estimates (q, u) on the confidential sample itself, per replication and
estimand. With the per-file table that is everything the q_org target needs,
so it can be evaluated after the fact on runs made with target "population".

Usage:
  python -m src.harness.run --config configs/oracle_proper.yaml --reps 0:1000
  python -m src.harness.run --config ... --reps 0:1000 --chunk-size 100
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from src.combining_rules.rules import naive, pool
from src.config import ExperimentConfig
from src.data.simulation import SimulationDGP
from src.evaluation.estimands import estimands_fast
from src.generators.oracle import ParametricOracle
from src.results_io import chunk_done, chunk_path, files_path, orig_path, write_chunk
from src.seeds import derive_seed

ESTIMAND_NAMES = ("E1", "E2", "E3")


def make_generator_by_name(name: str, kwargs: dict):
    """Registry. Deep-generator modules are imported lazily so the oracle runs
    without torch or R installed."""
    if name == "oracle":
        return ParametricOracle(**kwargs)
    if name in ("ctgan", "tvae"):
        from src.generators.ctgan_generators import CTGANGenerator, TVAEGenerator
        return (CTGANGenerator if name == "ctgan" else TVAEGenerator)(**kwargs)
    if name == "synthpop":
        from src.generators.synthpop_generator import SynthpopGenerator
        return SynthpopGenerator(**kwargs)
    if name == "tabpfn":
        from src.generators.tabpfn_generator import TabPFNGenerator
        return TabPFNGenerator(**kwargs)
    raise ValueError(f"unknown generator {name!r}")


def make_generator(cfg: ExperimentConfig):
    return make_generator_by_name(cfg.generator, cfg.generator_kwargs)


def make_dataset(cfg: ExperimentConfig):
    if cfg.dataset == "simulation":
        dgp = SimulationDGP()
        return dgp, estimands_fast, dgp.truth()
    if cfg.dataset == "acs":
        from src.data.acs import ACSPopulation, estimands_acs
        pop = ACSPopulation()                       # cached: loaded once per process
        return pop, estimands_acs, pop.truth()
    raise ValueError(f"unknown dataset {cfg.dataset!r}")


def _dispersion(files: list[pd.DataFrame], real: pd.DataFrame) -> dict:
    """Pre-flight: mean over files of Var(synthetic column) / Var(real column)."""
    real_var = real.var()
    ratios = pd.concat([f.var() / real_var for f in files], axis=1).mean(axis=1)
    return {f"disp_{c}": float(v) for c, v in ratios.items()}


def _rows(base: dict, pooled_or_interval, targets: np.ndarray, j: int) -> list[dict]:
    """Flatten one estimand's intervals into result rows."""
    out = []
    if hasattr(pooled_or_interval, "intervals"):
        p = pooled_or_interval
        extra = {"b": p.b, "u_bar": p.u_bar, "kappa": p.kappa, "files": p.m}
        intervals = p.intervals.values()
    else:
        extra = {"b": np.nan, "u_bar": pooled_or_interval.variance,
                 "kappa": np.nan, "files": 1}
        intervals = [pooled_or_interval]
    for iv in intervals:
        out.append({**base, **extra,
                    "estimand": ESTIMAND_NAMES[j], "estimator": iv.estimator,
                    "target_value": targets[j], "estimate": iv.estimate,
                    "variance": iv.variance, "df": iv.df,
                    "lo": iv.lo, "hi": iv.hi, "width": iv.width,
                    "covered": iv.covers(targets[j]), "corrected": iv.corrected})
    return out


def _file_rows(rep, step, Q, U, n_rej) -> list[dict]:
    """One row per file and estimand: the per-file estimate q and its variance
    estimate u, the inputs every combining rule starts from."""
    return [{"rep": rep, "step": step, "file": i, "estimand": ESTIMAND_NAMES[j],
             "q": float(Q[i, j]), "u": float(U[i, j]), "n_rejected": int(n_rej[i])}
            for i in range(Q.shape[0]) for j in range(Q.shape[1])]


def _pool_step(step, files, real, analyse, targets, cfg, base, t_fit, t_gen, n_rej,
               file_rows=None):
    Q, U = zip(*(analyse(f) for f in files))
    Q, U = np.array(Q), np.array(U)
    if file_rows is not None:
        file_rows += _file_rows(base["rep"], step, Q, U, n_rej)
    base = {**base, "step": step, "runtime_fit": t_fit, "runtime_gen": t_gen,
            "n_rejected": int(sum(n_rej)), **_dispersion(files, real)}
    rows = []
    for j in range(len(ESTIMAND_NAMES)):
        p = pool(Q[:, j], U[:, j], r=cfg.r, alpha=cfg.alpha, target=cfg.target,
                 estimators=cfg.estimators)
        rows += _rows(base, p, targets, j)
    return rows


def run_replication(cfg: ExperimentConfig, rep: int, file_rows: list | None = None,
                    orig_rows: list | None = None) -> list[dict]:
    """Result rows for one replication. If `file_rows` is a list, the per-file
    estimates (q, u) of every step are appended to it as well; if `orig_rows`
    is a list, the estimates on the confidential sample itself."""
    dgp, analyse, truth = make_dataset(cfg)
    real = dgp.sample(cfg.n_org, np.random.default_rng(derive_seed(cfg.seed0, rep, 0, "data")))
    q_org, u_org = analyse(real)
    if orig_rows is not None:
        orig_rows += [{"rep": rep, "estimand": ESTIMAND_NAMES[j], "q": float(q_org[j]),
                       "u": float(u_org[j])} for j in range(len(ESTIMAND_NAMES))]
    targets = (np.array([truth[e] for e in ESTIMAND_NAMES])
               if cfg.target == "population" else q_org)
    base = {"rep": rep}
    rows: list[dict] = []

    # Steps 1 and 2: one fit, m files
    if {1, 2} & set(cfg.steps):
        n_files = cfg.m if 2 in cfg.steps else 1
        t0 = time.perf_counter()
        gen = make_generator(cfg).fit(real, train_seed=derive_seed(cfg.seed0, rep, 2, "train"))
        t_fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        files, n_rej = [], []
        for i in range(n_files):
            files.append(gen.generate(cfg.n_syn, gen_seed=derive_seed(cfg.seed0, rep, 2, "gen", i)))
            n_rej.append(_rejected(gen))
        t_gen = time.perf_counter() - t0

        if 1 in cfg.steps:
            q, u = analyse(files[0])
            if file_rows is not None:
                file_rows += _file_rows(rep, 1, np.atleast_2d(q), np.atleast_2d(u), n_rej[:1])
            b1 = {**base, "step": 1, "runtime_fit": t_fit, "runtime_gen": t_gen / n_files,
                  "n_rejected": n_rej[0], **_dispersion(files[:1], real)}
            for j in range(len(ESTIMAND_NAMES)):
                rows += _rows(b1, naive(q[j], u[j], cfg.alpha), targets, j)
        if 2 in cfg.steps:
            rows += _pool_step(2, files, real, analyse, targets, cfg, base, t_fit, t_gen, n_rej,
                               file_rows)

    # Steps 3 and 4: K fits, one file each
    for step in (3, 4):
        if step not in cfg.steps:
            continue
        files, n_rej, t_fit, t_gen = [], [], 0.0, 0.0
        for k in range(cfg.K):
            if step == 4:
                boot_seed = derive_seed(cfg.seed0, rep, 4, "boot", k)
                data_k = real.sample(n=cfg.n_org, replace=True,
                                     random_state=boot_seed).reset_index(drop=True)
            else:
                data_k = real
            t0 = time.perf_counter()
            gen = make_generator(cfg).fit(
                data_k, train_seed=derive_seed(cfg.seed0, rep, step, "train", k))
            t_fit += time.perf_counter() - t0
            t0 = time.perf_counter()
            files.append(gen.generate(
                cfg.n_syn, gen_seed=derive_seed(cfg.seed0, rep, step, "gen", k)))
            n_rej.append(_rejected(gen))
            t_gen += time.perf_counter() - t0
        rows += _pool_step(step, files, real, analyse, targets, cfg, base, t_fit, t_gen, n_rej,
                           file_rows)

    return rows


def _rejected(gen) -> int:
    """Rows the generator rejected in its last generate() call (TabPFN only)."""
    return int(getattr(gen, "last_n_rejected", 0))


def provenance() -> dict:
    """Where and how a chunk ran. Results are reproducible given the config and
    seeds only on the same kind of hardware and thread count: floating-point
    sums in a different order change a neural net's training path."""
    threads = os.environ.get("OMP_NUM_THREADS", "")
    gpu = ""
    if "torch" in sys.modules:                     # only if a generator loaded it
        import torch
        threads = str(torch.get_num_threads())
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    return {"node": platform.node(), "threads": threads, "gpu": gpu}


def code_version() -> str:
    """Git commit of the code that produced a row; '-dirty' if uncommitted changes.

    The config hash identifies the design, not the code. Stamping the commit is
    what lets you see that results from before and after a code change are mixed.
    """
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run(cfg: ExperimentConfig, start: int, stop: int, chunk_size: int) -> None:
    h = cfg.hash()
    version = code_version()
    print(f"{cfg.name}  hash={h}  reps {start}:{stop}  chunk={chunk_size}", flush=True)
    for c0 in range(start, stop, chunk_size):
        c1 = min(c0 + chunk_size, stop)
        path = chunk_path(cfg.name, h, c0, c1)
        if chunk_done(path):
            print(f"  {path.name}  exists, skipped", flush=True)
            continue
        t0 = time.perf_counter()
        rows, file_rows, orig_rows = [], [], []
        for rep in range(c0, c1):
            rows += run_replication(cfg, rep, file_rows, orig_rows)
        prov = provenance()
        for row in rows:
            row["config_hash"] = h
            row["code_version"] = version
            row.update(prov)
        for row in file_rows + orig_rows:
            row["config_hash"] = h
        # Per-file values first, the chunk last: an existing chunk is what marks
        # the range as done, so it always has its per-file companions.
        write_chunk(orig_rows, orig_path(cfg.name, h, c0, c1))
        write_chunk(file_rows, files_path(cfg.name, h, c0, c1))
        write_chunk(rows, path)
        print(f"  {path.name}  {time.perf_counter() - t0:6.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", default="0:100", help="start:stop, stop exclusive")
    ap.add_argument("--chunk-size", type=int, default=100)
    args = ap.parse_args()
    start, stop = (int(x) for x in args.reps.split(":"))
    run(ExperimentConfig.from_yaml(args.config), start, stop, args.chunk_size)


if __name__ == "__main__":
    main()
