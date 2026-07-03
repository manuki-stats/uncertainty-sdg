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

Python 3.10 in the conda env `synth`. Pinned in `requirements.txt` (torch 2.2.2,
opacus 1.4.0, synthcity 0.2.12, rpy2 3.6.7 --- these versions are load-bearing;
synthcity requires torch < 2.3).

```
conda create -n synth python=3.10 -y
conda activate synth
pip install -r requirements.txt
```

**R bridge (Windows).** synthpop runs in native R (4.5.1) via rpy2. rpy2 must
find R: set `R_HOME` and put R's `bin\x64` on PATH. This is wired into the conda
env via an activation script:
`%CONDA_PREFIX%\etc\conda\activate.d\rpy2_env.bat`, so `conda activate synth`
sets it automatically. R must have `synthpop` installed (`install.packages("synthpop")`).

## Build order (the important part)

1. **Harness + oracle first --- not the generators.** Implement the evaluation
   loop and the fully-synthetic combining rules, then the parametric oracle.
2. **The gate:** the oracle must give ~95% coverage on E1--E3. If it does, the
   combining-rules code is correct. If not, fix it before any neural net.
3. Add synthpop (verify PROPER, Bayesian-bootstrap synthesis) and CTGAN.
4. Steps 1--3 of the study: naive baseline, multiple synthesis, DGE.
