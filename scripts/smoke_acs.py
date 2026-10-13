"""Smoke test of every generator on the real-data coding: one small replication each.

Usage (repo root, env synth2):  python scripts/smoke_acs.py [--file-only] [name ...]
    names: synthpop_plugin synthpop_proper tabpfn tvae ctgan   (default: all five)
    --file-only: skip the replication (TabPFN on a laptop CPU needs ~45 min for it)

For each generator: a reduced cell (n_org 500, m = K = 2; CTGAN and TVAE at 2 epochs on
CPU, from the tuning configs since their cell configs wait for the tuning winners), then
  1. one fit and one 1000-row file: columns, finiteness and category codes checked;
  2. one replication of Steps 1-4 through the real run_replication.
Checks the wiring, not the generators' quality. Writes nothing.
"""
from __future__ import annotations

import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                                     # noqa: E402
import pandas as pd                                                    # noqa: E402
import yaml                                                            # noqa: E402

from src.config import ExperimentConfig                                # noqa: E402
from src.data.acs import DISCRETE, LEVELS, ORDER, ACSPopulation        # noqa: E402
from src.harness import run as run_mod                                 # noqa: E402

CELLS = {"synthpop_plugin": "configs/acs_synthpop_plugin.yaml",
         "synthpop_proper": "configs/acs_synthpop_proper.yaml",
         "tabpfn": "configs/acs_tabpfn.yaml"}
TUNED = {"tvae": "configs/tune_tvae_acs.yaml", "ctgan": "configs/tune_ctgan_acs.yaml"}


def smoke_config(key: str) -> ExperimentConfig:
    if key in CELLS:
        cfg = ExperimentConfig.from_yaml(CELLS[key])
    else:
        t = yaml.safe_load(Path(TUNED[key]).read_text())
        kw = {**t["generator_kwargs"], "enable_gpu": False, "params": {"epochs": 2}}
        cfg = ExperimentConfig(name=t["name"], generator=t["generator"], generator_kwargs=kw,
                               dataset="acs", estimators=("T_alt", "T_s", "T_s_ppd"))
    return replace(cfg, name=cfg.name + "_smoke", n_org=500, m=2, K=2)


def check_file(cfg: ExperimentConfig, pop: ACSPopulation) -> None:
    real = pop.sample(cfg.n_org, np.random.default_rng(123))
    t0 = time.perf_counter()
    gen = run_mod.make_generator(cfg).fit(real, train_seed=1)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    syn = gen.generate(1000, gen_seed=2)
    t_gen = time.perf_counter() - t0
    assert list(syn.columns) == list(ORDER), f"columns {list(syn.columns)}"
    assert len(syn) == 1000, f"{len(syn)} rows"
    assert np.all(np.isfinite(syn.to_numpy(dtype=float))), "non-finite values"
    for c in DISCRETE:
        bad = set(syn[c].unique()) - {float(i) for i in range(len(LEVELS[c]))}
        assert not bad, f"{c}: codes outside 0..{len(LEVELS[c]) - 1}: {sorted(bad)[:5]}"
    print(f"  file check OK  (fit {t_fit:.1f}s, 1000 rows {t_gen:.1f}s)")
    if getattr(gen, "inferred_categorical", None):
        print(f"    library-inferred categoricals overridden: {gen.inferred_categorical}")
    print(f"    mean log_wage {syn.log_wage.mean():.3f} (real {real.log_wage.mean():.3f}); "
          f"female {np.mean(syn.sex == 1):.3f} (real {np.mean(real.sex == 1):.3f}); "
          f"age {syn.age.min():.0f}-{syn.age.max():.0f}; hours {syn.hours.min():.0f}-{syn.hours.max():.0f}")


def check_replication(cfg: ExperimentConfig) -> None:
    t0 = time.perf_counter()
    file_rows, orig_rows = [], []
    rows = pd.DataFrame(run_mod.run_replication(cfg, 0, file_rows, orig_rows))
    dt = time.perf_counter() - t0
    assert set(rows.step) == {1, 2, 3, 4}, f"steps {sorted(set(rows.step))}"
    assert len(orig_rows) == 3 and len(file_rows) == 3 * (1 + cfg.m + 2 * cfg.K)
    view = rows[rows.estimator.isin(["naive", "T_alt"])].pivot_table(
        index="step", columns="estimand", values="estimate")
    print(f"  replication OK ({dt:.1f}s); T_alt / naive estimates by step "
          f"(truth {rows.groupby('estimand').target_value.first().round(4).to_dict()}):")
    print("    " + view.round(4).to_string().replace("\n", "\n    "))


def main(keys: list[str], file_only: bool = False) -> int:
    pop = ACSPopulation()
    failed = []
    for key in keys:
        cfg = smoke_config(key)
        print(f"\n=== {key}: {cfg.generator} {cfg.generator_kwargs}")
        try:
            check_file(cfg, pop)
            if not file_only:
                check_replication(cfg)
        except Exception:
            traceback.print_exc()
            failed.append(key)
    print(f"\nsmoke: {len(keys) - len(failed)}/{len(keys)} passed"
          + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    only = "--file-only" in args
    names = [a for a in args if a != "--file-only"]
    sys.exit(main(names or list(CELLS) + list(TUNED), file_only=only))
