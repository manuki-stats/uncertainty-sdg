# Uncertainty propagation in deep generative models for synthetic tabular data

Code, configurations and results of the master's thesis *Uncertainty propagation in
deep generative models for synthetic tabular data* (Manuel Soffici, LMU Munich, MSc
Statistics and Data Science, 2026; supervisor Prof. Dr. Jörg Drechsler, IAB).

The thesis asks whether deep generators for tabular data (CTGAN, TVAE, TabPFN-3.5)
propagate uncertainty well enough for valid inference from fully synthetic releases.
It compares four release designs (one file; m files from one fit; seed-varying
refits; bootstrap refits) on a simulation with known truth and on a pseudo-population
built from the American Community Survey, with `synthpop`-CART as positive control
and a correctly specified parametric oracle as the check on the pipeline.

This README is the electronic appendix of the thesis.

## Contents

| Location | Contents |
|---|---|
| `thesis.pdf`, `slides.pdf` | The thesis and the slides of its defence. |
| `src/` | The Python package: generator wrappers (`generators/`), combining rules (`combining_rules/`), estimands, metrics and the pMSE ratio (`evaluation/`), data access (`data/`), and the experimental harness with the run loop, the tuning protocol and the pre-flight checks (`harness/`). |
| `configs/` | One YAML file per experimental cell and per tuning run. A cell's results are named by the hash of its configuration. |
| `slurm/` | Job scripts for the LRZ AI Systems cluster: the array job, the submission helper, and the benchmark and tuning jobs. |
| `scripts/` | Construction of the ACS pseudo-population from the public microdata and the checks of the analysis model; the scripts that build every figure and appendix table; the TabPFN and CTGAN diagnostics. |
| `extracts/acs2024_us/` | The ACS pseudo-population, the tuning sample, the population values of E1–E3, and a manifest with the filters and checksums. |
| `results/summary/` | The summary of every cell for both targets, from which every table and figure of the simulation and real-data chapters is built. |
| `results/tuning/` | Logs and selected configurations of both tuning rounds. |
| `docs/` | The pre-registration of the real-data study with its amendments, the scorecard of all pre-registered predictions, and the exact package versions. |
| `tests/` | Unit tests, including the oracle gate. |

### The package

```
src/
  config.py               one YAML config fully describes a cell; its hash names the results
  seeds.py                independent seed streams for numpy, torch, R and each generator
  results_io.py           chunked raw results (one CSV per range of replications)
  combining_rules/rules.py  T_alt, T_s, T_s(PPD), T_f; population and q_org targets
  data/simulation.py      the data-generating process of the simulation
  data/acs.py             the ACS pseudo-population and its analysis model
  evaluation/estimands.py E1 (mean), E2 (main slope), E3 (interaction)
  evaluation/metrics.py   coverage with Wilson intervals, the escalation rule, RMSE
  evaluation/pmse.py      pMSE ratio with the fixed propensity model
  evaluation/summarise.py one summary CSV per cell (and per target)
  generators/             oracle, synthpop (native R via rpy2), CTGAN and TVAE (ctgan 0.12.1),
                          TabPFN-3.5 (tabpfn-extensions 0.6.2); training and generation seeded separately
  harness/run.py          the evaluation loop: one replication, Steps 1-4
  harness/tune.py         equal-budget two-stage tuning, scored by pMSE, never by coverage
  harness/check_generator.py  pre-flight checks before any coverage is computed
  harness/progress.py     progress of a running cell
```

## Environment

One conda environment, `synth2`, holds everything: Python 3.10, R 4.5.1, `synthpop`
and `rpy2` from conda-forge (`environment.yml`), and the Python packages from pip
(`requirements.txt`). `environment-lock.yml` and `docs/environment_versions.txt`
record the exact versions used for the results.

```
mamba env create -f environment.yml
conda activate synth2
pip install --no-cache-dir -r requirements.txt
python scripts/env_check.py      # versions, torch/CUDA, the R bridge
pytest -q                        # unit tests, including the oracle gate
```

**Windows.** If R comes from a system install, rpy2 must find it: set `R_HOME` and put
R's `bin\x64` on PATH (an activation script in `%CONDA_PREFIX%\etc\conda\activate.d\`
does this), and install `synthpop` in that R.

## Reproducing the results

Every result is regenerated from a committed configuration and a logged seed. Raw
results are written in chunks to `results/raw/` (git-ignored); every row records the
code version, node, thread count and GPU, because neural fits can depend on them.

**One cell, locally.** The oracle needs neither a GPU nor R, so it is the quickest cell to run:

```
python -m src.harness.run --config configs/oracle_plugin.yaml --reps 0:100 --chunk-size 10
python -m src.evaluation.summarise --config configs/oracle_plugin.yaml
```

**On the cluster.** Each cell runs as one SLURM array job; resubmitting a range runs
only the missing chunks.

```
slurm/submit.sh configs/<cell>.yaml 0:1000 cpu --chunk 10 --workers 8 --threads 2 --time 12:00:00
python -m src.harness.progress configs/<cell>.yaml 0:1000 --chunk-size 10
```

Profiles: `cpu` (lrz-cpu), `gpu` and `gpu-mcml` (one H100). Keep `--chunk` fixed for a
cell. Do not change the checked-out code while a cell is queued: tasks run whatever
code is checked out when they start.

**Summaries.** One CSV per cell in `results/summary/`, named `<cell>__<config hash>.csv`.
For the real-data cells, the second target (each replication's original-data
estimate) is rebuilt from the same runs:

```
python -m src.evaluation.summarise --config configs/acs_tvae.yaml
python -m src.evaluation.summarise --config configs/acs_tvae.yaml --target sample   # __qorg.csv
```

**Tuning.** Run once per generator and data source, before any experiment:

```
python -m src.harness.tune --config configs/tune_tvae.yaml --smoke   # minutes
python -m src.harness.tune --config configs/tune_tvae.yaml
```

Output goes to `results/tuning/<name>__<hash>/`, with `winner.yaml` holding the
selected settings.

**The ACS pseudo-population.** Built from the public 2024 ACS 1-year PUMS, downloaded
by the script and checked against the checksums in the manifest:

```
python scripts/build_acs_extract.py     # population and tuning sample
python scripts/acs_analysis_model.py    # analysis model checks, design-based check, true values
```

**Figures and appendix tables**, from the committed summaries:

```
python scripts/fig_sim_coverage_pred.py --summary results/summary --out figures/
python scripts/fig_var_ratio_box.py --out figures/
python scripts/make_appendix_tables.py --summary results/summary --out appendix_tables/
```

The boxplot of per-run variance ratios reads `results/summary/sim_var_ratio_runs.csv`,
which `fig_var_ratio_box.py --raw results/raw --save` writes from the raw results.

## Run times

Per replication of Steps 1–4 ($m = K = 10$, 21 fits and 30 files), on the LRZ AI
Systems cluster. CPU times are for one process; GPU cells ran four processes per H100.

| Cell | Simulation | ACS | Replications |
|---|---|---|---|
| Oracle (four modes) | short, CPU only | — | 1,000 each |
| `synthpop`, each mode | 15–18 s | ~46 s | 1,000 |
| TVAE (CPU) | 21–50 min | 7–13 h | 1,000 |
| CTGAN (H100) | ~47 GPU-min | ~1.6 GPU-h | 100 |
| TabPFN-3.5 (H100) | ~10 GPU-min | ~25 GPU-min | 1,000 |

Run times vary with the node, and on lrz-cpu with how many jobs share a core.

## Reproducibility notes

- Code changes made while experiments were running were verified to reproduce stored
  replications bit for bit, and the final code rebuilds every committed summary
  identically.
- The generator policy (thesis, Chapter 4): generators are used as shipped, except
  where they contradict their documented behaviour or return invalid values. The
  fixes are CTGAN's category frequencies at generation, CTGAN's GPU seeding, TabPFN's
  categorical declaration on the ACS data, and the redraw of non-finite TabPFN values.
- The dequantised TabPFN cell (`configs/acs_tabpfn_dequant.yaml`) is an exploratory
  sensitivity analysis; the option is off by default, and with it off the code
  rebuilds the as-shipped files exactly.

## Licence

MIT, see `LICENSE`. The ACS microdata are public domain (U.S. Census Bureau).
