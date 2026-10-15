"""Equal-budget hyperparameter tuning, scored by pMSE, never by coverage.

Protocol (identical for every tuned generator):

  Stage 1  the default configuration plus `n_random` random draws from the
           search space; one training fit each, scored as the mean pMSE ratio
           over `n_files` generated files at r = 1.
  Stage 2  the default plus the `n_top` best non-default configurations,
           each refitted with `n_seeds` fresh training seeds. Stage-1 scores
           are discarded, because the minimum of many noisy scores is biased
           towards lucky seeds.
  Select   the one-standard-error rule (Breiman et al. 1984; Hastie,
           Tibshirani & Friedman 2009, sec. 7.10): among the finalists whose
           stage-2 mean lies within one standard error of the best mean, take
           the cheapest to fit (median stage-2 fit time). With three seeds per
           configuration the means are often indistinguishable, and paying
           several times the compute of Steps 3-4 for an indistinguishable
           score is poor value. The best-by-mean configuration is reported
           alongside. The rule was fixed on 23 Sept 2026, after the TVAE run
           (whose outcome it does not change) and before CTGAN's stage 2.

With n_random = 29, n_top = 3, n_seeds = 3 that is 42 fits per generator. By
Bergstra & Bengio's (2012) argument, 29 independent draws put at least one in
the best 10% of the space with probability 1 - 0.9^29 = 0.95.

Tuning runs on its own sample, drawn with its own seed0, so it is independent
of every experiment replication. On the real data that sample is the ACS
tuning sample, reserved from the pool before the population was formed
(extracts/acs2024_us/d_tune.csv); the propensity regressors are M2's monomials
(src/data/acs.py, FEATURE_FORMULA), and there is no correctly specified oracle
to score as a reference. It happens once, at n_org, before any
experiment: the best epochs and batch size depend on n.

The run also reports three diagnostics, so the procedure states its own limits:
  - the correctly specified plug-in oracle scored by the same protocol on the
    same sample (simulation only): how far the winner is from a good fit;
  - the winner's stage-2 margin over the default, with its standard error:
    whether tuning bought anything beyond seed noise;
  - search-space edges the winner sits on: whether a bound or cap was binding.

Every fit is written to its own JSON file as soon as it finishes, so a killed
run resumes where it stopped.

Parallel workers (slurm/tune.sbatch). `--worker 1` and `--worker 2` run one
stage as one of several processes sharing a directory: each process claims a
fit by creating fit_*.claim atomically, computes it, and moves on, so the fits
spread over the workers by load. A claim belongs to the Slurm job that made it;
a claim left by an earlier, killed job is taken over. A stage-2 worker exits if
stage 1 is incomplete. The final plain run loads everything, computes any fit
still missing (a worker that died mid-fit), selects the winner and writes
winner.yaml; with --no-retry-failed it keeps failed fits as failed rather than
refitting them in the same job. Each fit's seeds depend only on (seed0, stage,
trial, seed index), so the split across workers does not change any fit.

Usage:
  python -m src.harness.tune --config configs/tune_tvae.yaml
  python -m src.harness.tune --config configs/tune_tvae.yaml --smoke
  python -m src.harness.tune --config configs/tune_tvae.yaml --worker 1   # see above
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.simulation import FORMULA, SimulationDGP
from src.evaluation.pmse import DerivedMomentFeatures, MomentFeatures, pmse_score
from src.seeds import derive_seed

TUNE_ROOT = Path("results/tuning")


# ---------------------------------------------------------------- config

@dataclass(frozen=True)
class TuneConfig:
    name: str
    generator: str                          # registry name, e.g. "ctgan"
    space: dict                             # parameter -> sampling spec
    generator_kwargs: dict = field(default_factory=dict)   # fixed, non-tuned kwargs
    dataset: str = "simulation"
    formula: str = FORMULA                  # analysis model defining the propensity regressors
    n_tune: int = 5000                      # = n_org of the experiments
    n_random: int = 29
    n_files: int = 3
    n_top: int = 3
    n_seeds: int = 3
    seed0: int = 20260923                   # distinct from every experiment seed0
    force_params: dict = field(default_factory=dict)       # smoke tests only

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=list)
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TuneConfig":
        return cls(**yaml.safe_load(Path(path).read_text()))

    def smoke(self) -> "TuneConfig":
        """A minutes-long run of the full pipeline: tiny sample, few fits, 2 epochs."""
        return replace(self, name=self.name + "_smoke", n_tune=500, n_random=2,
                       n_files=1, n_top=1, n_seeds=2, force_params={"epochs": 2})


# ---------------------------------------------------------------- search space

def sample_params(space: dict, rng: np.random.Generator) -> dict:
    """One configuration. Specs:
        {choice: [a, b, c]}        uniform over the listed values
        {loguniform: [lo, hi]}     log-uniform on [lo, hi], 3 significant digits
        {layers: {width: [...], depth: [...]}}   e.g. [256, 256, 256]
    """
    out = {}
    for name in sorted(space):                  # sorted: order-independent of the YAML
        (kind, arg), = space[name].items()
        if kind == "choice":
            out[name] = _py(arg[rng.integers(len(arg))])
        elif kind == "loguniform":
            lo, hi = float(arg[0]), float(arg[1])
            out[name] = float(f"{np.exp(rng.uniform(np.log(lo), np.log(hi))):.3g}")
        elif kind == "layers":
            w = _py(arg["width"][rng.integers(len(arg["width"]))])
            d = _py(arg["depth"][rng.integers(len(arg["depth"]))])
            out[name] = [w] * d
        else:
            raise ValueError(f"unknown spec {kind!r} for {name!r}")
    return out


def _py(v):
    return v.item() if hasattr(v, "item") else v


def edges(params: dict, space: dict) -> list[str]:
    """Parameters sitting on an edge of their range (numeric choices at the
    extreme value, log-uniform draws within 10% of the log-range of a bound)."""
    flags = []
    for name, value in params.items():
        if name not in space:
            continue
        (kind, arg), = space[name].items()
        if kind == "choice" and all(isinstance(a, (int, float)) for a in arg) and len(arg) > 1:
            if value == max(arg):
                flags.append(f"{name}={value} (max)")
            elif value == min(arg):
                flags.append(f"{name}={value} (min)")
        elif kind == "loguniform":
            lo, hi = np.log(float(arg[0])), np.log(float(arg[1]))
            t = (np.log(value) - lo) / (hi - lo)
            if t < 0.1 or t > 0.9:
                flags.append(f"{name}={value} ({'low' if t < 0.1 else 'high'} end)")
        elif kind == "layers":
            if len(value) == max(arg["depth"]) and value[0] == max(arg["width"]):
                flags.append(f"{name}={value} (largest)")
    return flags


# ---------------------------------------------------------------- evaluation

def _default_factory(generator: str, kwargs: dict):
    from src.harness.run import make_generator_by_name
    return make_generator_by_name(generator, kwargs)


def _fit_path(out: Path, stage: str, trial: int, seed_idx: int) -> Path:
    return out / f"fit_{stage}_t{trial:02d}_s{seed_idx}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a uniquely named temporary file and an atomic rename: parallel
    workers share the output directory. Windows refuses a rename onto a file that
    another process is renaming at the same moment, so that case is retried."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
    os.remove(tmp)
    raise PermissionError(f"could not replace {path}")


def _write_json(obj: dict, path: Path) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=1))


def evaluate_fit(cfg: TuneConfig, data: pd.DataFrame, features: MomentFeatures,
                 generator: str, kwargs: dict, stage_no: int, trial: int, seed_idx: int,
                 factory) -> dict:
    """One training fit, n_files generated files, their pMSE ratios."""
    rec = {"generator": generator, "kwargs": kwargs, "trial": trial, "seed_idx": seed_idx}
    try:
        t0 = time.perf_counter()
        gen = factory(generator, kwargs).fit(
            data, train_seed=derive_seed(cfg.seed0, trial, stage_no, "train", seed_idx))
        rec["fit_s"] = time.perf_counter() - t0
        ratios = []
        t0 = time.perf_counter()
        for f in range(cfg.n_files):
            syn = gen.generate(len(data), gen_seed=derive_seed(
                cfg.seed0, trial, stage_no, "gen", seed_idx * 100 + f))
            ratios.append(pmse_score(data, syn, features).ratio)
        rec["gen_score_s"] = time.perf_counter() - t0
        rec["ratios"] = ratios
        rec["score"] = float(np.mean(ratios))
        rec["error"] = None
    except (ImportError, MemoryError, KeyboardInterrupt):
        raise                               # a broken environment stops the run, not one config
    except Exception as e:                  # a crashed configuration loses, it doesn't stop the run
        rec.update(ratios=[], score=float("inf"), error=f"{type(e).__name__}: {e}")
    return rec


def _run_or_load(path: Path, fn, retry_failed: bool = True) -> dict:
    """Load a finished fit from disk, or run it. Failed fits are retried on the
    next run rather than reloaded, so fixing the cause needs no manual cleanup
    (unless retry_failed is False: the parallel job's final pass)."""
    if path.exists():
        rec = json.loads(path.read_text())
        if rec.get("error") is None or not retry_failed:
            return rec
    rec = fn()
    _write_json(rec, path)
    return rec


def _claim(path: Path) -> bool:
    """Claim the fit at `path` for this process (worker mode). The claim file
    holds "<slurm job>:<pid>"; a claim from another job is stale and taken over."""
    claim = path.with_suffix(".claim")
    job = os.environ.get("SLURM_JOB_ID", "local")
    token = f"{job}:{os.getpid()}"
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            owner = claim.read_text().split(":")[0]
        except OSError:
            return False
        if owner in ("", job):                # being written, or a live worker of this job
            return False
        _atomic_write_text(claim, token)
        return claim.read_text() == token     # two takers: the last writer wins
    with os.fdopen(fd, "w") as f:
        f.write(token)
    return True


def _log(rec: dict, label: str) -> None:
    if rec["error"]:
        print(f"  {label}  FAILED  {rec['error']}", flush=True)
    else:
        print(f"  {label}  score {rec['score']:8.2f}   fit {rec['fit_s']:6.1f}s", flush=True)


# ---------------------------------------------------------------- driver

def run_tuning(cfg: TuneConfig, factory=_default_factory, root: Path = TUNE_ROOT,
               worker_stage: int | None = None, retry_failed: bool = True) -> dict | None:
    out = root / f"{cfg.name}__{cfg.hash()}"
    out.mkdir(parents=True, exist_ok=True)
    cfg_text = yaml.safe_dump(asdict(cfg), sort_keys=False)
    if not (out / "config.yaml").exists() or (out / "config.yaml").read_text() != cfg_text:
        _atomic_write_text(out / "config.yaml", cfg_text)
    if cfg.dataset == "simulation":
        data = SimulationDGP().sample(
            cfg.n_tune, np.random.default_rng(derive_seed(cfg.seed0, 0, 0, "tune_data")))
        features = MomentFeatures(cfg.formula).fit(data)
    elif cfg.dataset == "acs":
        from src.data.acs import FEATURE_FORMULA, derive, load_tuning_sample
        if cfg.formula != FEATURE_FORMULA:
            raise ValueError(f"acs tuning needs formula {FEATURE_FORMULA!r}, got {cfg.formula!r}")
        data = load_tuning_sample()
        if cfg.n_tune > len(data):
            raise ValueError(f"n_tune = {cfg.n_tune} but the tuning sample has {len(data)} rows")
        data = data.iloc[:cfg.n_tune].reset_index(drop=True)   # smoke runs only: a random subset
        features = DerivedMomentFeatures(cfg.formula, derive).fit(data)
    else:
        raise ValueError(f"unknown dataset {cfg.dataset!r}")

    rng = np.random.default_rng(derive_seed(cfg.seed0, 0, 0, "tune_config"))
    configs = [{}] + [sample_params(cfg.space, rng) for _ in range(cfg.n_random)]
    configs = [{**p, **cfg.force_params} for p in configs]

    def kw(params):
        return {**cfg.generator_kwargs, "params": params}

    print(f"{cfg.name}  hash={cfg.hash()}  n_tune={cfg.n_tune}  "
          f"stage 1: {len(configs)} fits, stage 2: {(cfg.n_top + 1) * cfg.n_seeds} fits",
          flush=True)
    if worker_stage is not None:
        _work(cfg, out, data, features, configs, kw, factory, worker_stage)
        return None

    # Stage 1
    s1 = []
    for t, params in enumerate(configs):
        rec = _run_or_load(_fit_path(out, "stage1", t, 0), lambda: evaluate_fit(
            cfg, data, features, cfg.generator, kw(params), 1, t, 0, factory), retry_failed)
        rec["params"] = params
        _log(rec, f"stage1 t{t:02d}{' (default)' if t == 0 else '          '}")
        s1.append(rec)

    finalists = _finalists(cfg, [r["score"] for r in s1])

    # Stage 2
    s2 = {}
    for t in finalists:
        recs = []
        for k in range(cfg.n_seeds):
            rec = _run_or_load(_fit_path(out, "stage2", t, k), lambda: evaluate_fit(
                cfg, data, features, cfg.generator, kw(configs[t]), 2, t, k, factory),
                retry_failed)
            _log(rec, f"stage2 t{t:02d} seed {k}")
            recs.append(rec)
        s2[t] = recs

    # Reference: the correct plug-in oracle under the stage-2 protocol (simulation only)
    ref = []
    for k in (range(cfg.n_seeds) if cfg.dataset == "simulation" else []):
        rec = _run_or_load(_fit_path(out, "reference", 0, k), lambda: evaluate_fit(
            cfg, data, features, "oracle", {"mode": "plugin"}, 3, 0, k, _default_factory))
        ref.append(rec)

    return _summarise(cfg, out, configs, s1, s2, ref)


def _finalists(cfg: TuneConfig, s1_scores: list[float]) -> list[int]:
    ranked = sorted(range(1, len(s1_scores)), key=lambda t: (s1_scores[t], t))
    return [0] + ranked[:cfg.n_top]


def _work(cfg, out, data, features, configs, kw, factory, stage: int) -> None:
    """Worker mode: compute this stage's unfinished, unclaimed fits, then return.
    A fit that exists (finished or failed) is never redone here."""
    if stage == 1:
        todo = [(t, 0) for t in range(len(configs))]
    elif stage == 2:
        scores = []
        for t in range(len(configs)):
            p = _fit_path(out, "stage1", t, 0)
            if not p.exists():
                print(f"stage 1 incomplete (t{t:02d} missing): this stage-2 worker exits; "
                      f"the final pass will fill the gap", flush=True)
                return
            scores.append(json.loads(p.read_text())["score"])
        todo = [(t, k) for t in _finalists(cfg, scores) for k in range(cfg.n_seeds)]
    else:
        raise ValueError(f"worker stage must be 1 or 2, got {stage}")
    for t, k in todo:
        path = _fit_path(out, f"stage{stage}", t, k)
        if path.exists() or not _claim(path):
            continue
        rec = evaluate_fit(cfg, data, features, cfg.generator, kw(configs[t]), stage, t, k,
                           factory)
        _write_json(rec, path)
        _log(rec, f"stage{stage} t{t:02d} seed {k}")


def _stats(recs: list[dict]) -> tuple[float, float]:
    scores = np.array([r["score"] for r in recs], dtype=float)
    if not np.all(np.isfinite(scores)):
        return float("inf"), float("nan")
    se = scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else float("nan")
    return float(scores.mean()), float(se)


def select_winner(table: list[dict]) -> tuple[dict, dict]:
    """One-standard-error rule. `table` rows need stage2_mean, stage2_se,
    fit_s and trial. Returns (selected, best_by_mean)."""
    rows = sorted(table, key=lambda r: (r["stage2_mean"], r["trial"]))
    best = rows[0]
    se = best["stage2_se"] if np.isfinite(best["stage2_se"]) else 0.0
    eligible = [r for r in rows if r["stage2_mean"] <= best["stage2_mean"] + se]
    selected = min(eligible, key=lambda r: (r["fit_s"], r["stage2_mean"], r["trial"]))
    return selected, best


def _summarise(cfg, out, configs, s1, s2, ref) -> dict:
    table = []
    for t, recs in s2.items():
        mean, se = _stats(recs)
        table.append({"trial": t, "default": t == 0, "stage1": s1[t]["score"],
                      "stage2_mean": mean, "stage2_se": se,
                      "fit_s": float(np.median([r.get("fit_s", np.nan) for r in recs])),
                      "params": configs[t]})
    table.sort(key=lambda r: (r["stage2_mean"], r["trial"]))
    if not np.isfinite(table[0]["stage2_mean"]):
        raise RuntimeError(
            f"no stage-2 finalist of {cfg.name} produced a valid score; no winner "
            f"written. See the fit_*.json files in {out}")
    win, best = select_winner(table)
    dflt = next(r for r in table if r["default"])
    ref_mean, ref_se = _stats(ref) if ref else (float("nan"), float("nan"))
    margin = dflt["stage2_mean"] - win["stage2_mean"]
    margin_se = float(np.hypot(win["stage2_se"], dflt["stage2_se"]))
    summary = {
        "name": cfg.name, "hash": cfg.hash(), "generator": cfg.generator,
        "selection_rule": "one-standard-error, cheapest within 1 SE of the best mean",
        "best_by_mean_trial": best["trial"], "best_by_mean_score": best["stage2_mean"],
        "best_by_mean_se": best["stage2_se"], "best_by_mean_fit_s": best["fit_s"],
        "winner_trial": win["trial"], "winner_is_default": win["default"],
        "winner_fit_s": win["fit_s"],
        "winner_params": win["params"],
        "winner_score": win["stage2_mean"], "winner_se": win["stage2_se"],
        "default_score": dflt["stage2_mean"], "default_se": dflt["stage2_se"],
        "margin_over_default": margin, "margin_se": margin_se,
        "margin_beyond_2se": bool(margin > 2 * margin_se) if not win["default"] else False,
        "reference_plugin_oracle": ref_mean, "reference_se": ref_se,
        "edges": edges(win["params"], cfg.space) if not win["default"] else [],
        "failed_fits": sum(r["error"] is not None for r in s1)
                       + sum(r["error"] is not None for rs in s2.values() for r in rs),
        "stage2_table": table,
    }
    _write_json(summary, out / "summary.json")
    (out / "winner.yaml").write_text(yaml.safe_dump(
        {"generator": cfg.generator,
         "generator_kwargs": {**cfg.generator_kwargs, "params": win["params"]}},
        sort_keys=False))
    _print_summary(summary, out)
    return summary


def _print_summary(s: dict, out: Path) -> None:
    print(f"\n=== {s['name']}: stage 2 (mean pMSE ratio over seeds, lower is better)")
    for r in s["stage2_table"]:
        tag = "default" if r["default"] else f"t{r['trial']:02d}"
        print(f"  {tag:8s} {r['stage2_mean']:8.2f} ± {r['stage2_se']:.2f}   "
              f"(stage 1: {r['stage1']:.2f}, fit {r['fit_s']:.0f}s)  {r['params']}")
    if np.isfinite(s["reference_plugin_oracle"]):
        print(f"  reference: plug-in oracle {s['reference_plugin_oracle']:.2f} "
              f"± {s['reference_se']:.2f}")
    else:
        print("  reference: none (no correctly specified oracle on real data)")
    bt = "default" if s["best_by_mean_trial"] == 0 else f"t{s['best_by_mean_trial']:02d}"
    wt = "default" if s["winner_is_default"] else f"t{s['winner_trial']:02d}"
    print(f"\nBest by mean: {bt} ({s['best_by_mean_score']:.2f} ± {s['best_by_mean_se']:.2f}, "
          f"fit {s['best_by_mean_fit_s']:.0f}s).  Selected by the 1-SE rule: {wt} "
          f"(fit {s['winner_fit_s']:.0f}s).")
    if s["winner_is_default"]:
        print("Winner: the DEFAULT configuration.")
    else:
        verdict = ("beyond seed noise" if s["margin_beyond_2se"]
                   else "NOT beyond seed noise (< 2 SE)")
        print(f"Winner: t{s['winner_trial']:02d}, {s['margin_over_default']:.2f} "
              f"± {s['margin_se']:.2f} below the default, {verdict}.")
        if s["edges"]:
            print("  On a search-space edge: " + "; ".join(s["edges"]))
    if s["failed_fits"]:
        print(f"  {s['failed_fits']} fit(s) failed; see the fit_*.json files.")
    print(f"\nFrozen settings: {out / 'winner.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny sample, 2 epochs, 5 fits: checks the pipeline, not the tuning")
    ap.add_argument("--worker", type=int, choices=[1, 2], default=None,
                    help="run one stage as one of several parallel workers (slurm/tune.sbatch)")
    ap.add_argument("--no-retry-failed", action="store_true",
                    help="keep failed fits as failed instead of refitting them")
    args = ap.parse_args()
    cfg = TuneConfig.from_yaml(args.config)
    s = run_tuning(cfg.smoke() if args.smoke else cfg, worker_stage=args.worker,
                   retry_failed=not args.no_retry_failed)
    if args.smoke and s is not None and s["failed_fits"]:
        raise SystemExit(f"smoke run: {s['failed_fits']} fit(s) failed")


if __name__ == "__main__":
    main()
