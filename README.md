# Uncertainty propagation & CI coverage in deep generative models for tabular data

Master's thesis (LMU Statistics & Data Science; supervisor: Prof. Dr. Jörg Drechsler, IAB).

**Question.** When deep generative models produce synthetic tabular data, do they
propagate uncertainty correctly for valid downstream inference --- or do they treat
generation like prediction, giving intervals that are too narrow (and sometimes
biased)? We diagnose this under full synthesis, then test whether multiple synthesis
and Deep Generative Ensembles (DGE) restore nominal 95% confidence-interval coverage.

## Layout

```
src/
  seeds.py                 seed numpy / torch / R from one place
  config.py                one config fully describes an experiment cell
  results_log.py           append-only flat CSV; idempotent, resumable
  combining_rules/
    fully_synthetic.py     RRR-2003 estimator + negative-variance correction (main)
    partial_synthetic.py   Reiter-2003 rules (optional Appendix E only)
  generators/
    base.py                the Generator interface (fit / generate)
    synthcity_generators.py  CTGAN, TVAE, TabDDPM (= "ddpm")
    synthpop_generator.py    native R synthpop via rpy2 --- the trusted control
    oracle.py                parametric oracle --- validates the harness
  evaluation/
    estimands.py           E1 mean, E2 main coefficient, E3 hard estimand
    metrics.py             coverage, width, bias; RMSE as the foil
  harness/
    run.py                 the evaluation loop --- the reusable core
configs/                   one YAML per experiment (start: oracle_sim.yaml)
results/                   results.csv (git-ignored) + its README
tests/                     test_oracle_gate.py --- the ~95% coverage gate
notebooks/                 exploration only, never the source of truth
```

## Reproducibility contract

Every result is regenerable from a committed config + a logged seed. Never
hand-edit results. `set_all_seeds(seed)` seeds Python, numpy, torch, and R; the
seed is written with every result row. Runs are idempotent (a completed
`(config_hash, seed)` is skipped) and resumable.

## Environment

One conda env, `synth2`, holds everything: Python 3.10, R 4.5.1, synthpop and
rpy2 come from conda-forge (`environment.yml`); the Python packages come from pip
(`requirements.txt`). These files build the env on the LRZ cluster (Linux; the
PyPI `torch` wheel there is the CUDA 13.0 build). The laptop env (Windows, CPU
torch, system R) predates them; see the note below.

```
mamba env create -f environment.yml
conda activate synth2
pip install --no-cache-dir -r requirements.txt
python scripts/env_check.py      # versions, torch/CUDA, the R bridge (tiny syn() call)
pytest -q                        # 23 passed
```

`environment-lock.yml` records the exact versions resolved on the cluster
(`conda env export --no-builds`). synthcity and TabDDPM are no longer used.

**Windows.** If R comes from a system install instead of conda, rpy2 must find it:
set `R_HOME` and put R's `bin\x64` on PATH (an activation script in
`%CONDA_PREFIX%\etc\conda\activate.d\` does this), and install synthpop in that R.

## Running on the LRZ cluster

Each design cell runs as one SLURM array job; every array task runs several
chunks of replications in parallel, one process per chunk.

```
slurm/submit.sh configs/<cell>.yaml 0:100 cpu --chunk 10 --workers 8 --threads 2 --time 12:00:00
python -m src.harness.progress configs/<cell>.yaml 0:100 --chunk-size 10
slurm/pack_results.sh            # -> ~/results_latest.tgz, to copy to the laptop
```

Profiles: `cpu` (lrz-cpu), `gpu` and `gpu-mcml` (one H100). Resubmitting a range
only runs the missing chunks. Keep `--chunk` fixed for a cell. Do not `git pull`
while a cell is queued or running: tasks run whatever code is checked out when
they start (every row records it in `code_version`).

## Tuning

Deep generators are tuned once, before any experiment, by an equal-budget
two-stage random search (`src/harness/tune.py`, protocol in its docstring):
42 fits per generator, scored by the pMSE ratio (`src/evaluation/pmse.py`),
never by coverage. Tuning uses its own sample (own `seed0`) at n = n_org.

    python -m src.harness.tune --config configs/tune_tvae.yaml --smoke   # minutes
    python -m src.harness.tune --config configs/tune_tvae.yaml           # the real run

Output goes to `results/tuning/<name>__<hash>/`: one JSON per fit (resumable),
`summary.json`, and `winner.yaml` with the settings to copy into the
experiment configs' `generator_kwargs`.

## Build order (the important part)

1. **Harness + oracle first --- not the generators.** Implement the evaluation
   loop and the fully-synthetic combining rules, then the parametric oracle.
2. **The gate:** the oracle must give ~95% coverage on E1--E3. If it does, the
   combining-rules code is correct. If not, fix it before any neural net.
3. Add synthpop (verify PROPER, Bayesian-bootstrap synthesis) and CTGAN.
4. Steps 1--3 of the study: naive baseline, multiple synthesis, DGE.
