"""Rebuild the Step-2 files of one replication exactly as the harness does and
look inside them: which rows drive a file's slope estimates?

Motivation: in TabPFN pass 1, ~4% of Step-2 files have E2 (slope of x1 at x2=0)
20-120 within-file SEs below their replication's median, with E3 shifted up and
E1 untouched. Leading hypothesis: a few rows with x1 outside the training
range and y not following the line there (high leverage, large residual).
Alternatives this script can distinguish: extreme y at ordinary x1; many
moderately aberrant rows; a file-wide distortion with no influential rows.

Usage (on a GPU node, through slurm/pyjob.sbatch):
  python -m scripts.inspect_tabpfn_file --config configs/tabpfn.yaml --reps 26,81
(-m from the repo root, so that `src` is importable)

Files are generated in order 0..m-1 from the same fit, as in run_replication,
and every file's (q, u) is checked against the stored per-file values, so the
reconstruction is verified before anything is read into it. The described
files are saved to results/bench/ for further analysis; nothing is written to
results/raw.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

TRUE = {"b0": 1.0, "b1": 2.0, "b2": -0.5, "b3": 1.5}    # y = 1 + 2x1 - .5x2 + 1.5x1x2 + e


def design(df: pd.DataFrame) -> np.ndarray:
    x1, x2 = df["x1"].to_numpy(float), df["x2"].to_numpy(float)
    return np.column_stack([np.ones(len(df)), x1, x2, x1 * x2])


def ols_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row leverage, residual and Cook's distance for y ~ x1 + x2 + x1:x2,
    plus the true conditional mean and the row's deviation from it."""
    X, y = design(df), df["y"].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    Q, _ = np.linalg.qr(X)
    h = (Q ** 2).sum(axis=1)
    p, n = X.shape[1], len(y)
    s2 = e @ e / (n - p)
    cook = e ** 2 * h / (p * s2 * (1 - h) ** 2)
    mu = TRUE["b0"] + TRUE["b1"] * df.x1 + TRUE["b2"] * df.x2 + TRUE["b3"] * df.x1 * df.x2
    return df.assign(true_mean=mu, dev_from_truth=df.y - mu, leverage=h, resid=e, cooks_d=cook)


def coefs(df: pd.DataFrame) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design(df), df["y"].to_numpy(float), rcond=None)
    return beta                                       # b0, b1 (=E2), b2, b3 (=E3)


def describe_file(df: pd.DataFrame, real: pd.DataFrame, label: str) -> None:
    d = ols_diagnostics(df)
    lo, hi = real.x1.min(), real.x1.max()
    ylo, yhi = real.y.min(), real.y.max()
    out_x = ((d.x1 < lo) | (d.x1 > hi)).sum()
    out_y = ((d.y < ylo) | (d.y > yhi)).sum()
    b = coefs(df)
    print(f"\n=== {label}: E2 = {b[1]:.4f}, E3 = {b[3]:.4f}, mean y = {df.y.mean():.4f}")
    print(f"rows with x1 outside the training range [{lo:.2f}, {hi:.2f}]: {out_x}; "
          f"y outside [{ylo:.2f}, {yhi:.2f}]: {out_y}")
    print(f"x1 quantiles 0/0.1/1/50/99/99.9/100%: "
          f"{np.round(np.quantile(d.x1, [0, .001, .01, .5, .99, .999, 1]), 2)}")
    cols = ["x2", "x1", "y", "true_mean", "dev_from_truth", "leverage", "cooks_d"]
    print("top 8 rows by Cook's distance:")
    print(d.sort_values("cooks_d", ascending=False)[cols].head(8).round(3).to_string())
    print("top 5 rows by |y - true mean|:")
    print(d.reindex(d.dev_from_truth.abs().sort_values(ascending=False).index)[cols]
          .head(5).round(3).to_string())
    for g in (0.0, 1.0):
        sub = d[d.x2 == g]
        n_out = int(((sub.x1 < lo) | (sub.x1 > hi)).sum())
        slope = np.polyfit(sub.x1, sub.y, 1)[0]
        print(f"group x2={g:.0f}: n = {len(sub)}, rows with x1 out of range = {n_out}, "
              f"OLS slope = {slope:.4f} (true {TRUE['b1'] + TRUE['b3'] * g:.1f})")
    inside = df[(df.x1 >= lo) & (df.x1 <= hi)]
    bi = coefs(inside)
    print(f"  without the {len(df) - len(inside)} out-of-range rows: E2 = {bi[1]:.4f}, E3 = {bi[3]:.4f}")
    order = d.cooks_d.sort_values(ascending=False).index
    for k in (1, 3, 10, 30):
        bk = coefs(df.drop(index=order[:k]))
        print(f"  without top {k:>2} Cook's rows: E2 = {bk[1]:.4f}, E3 = {bk[3]:.4f}")


def main():
    import src.harness.run as H                       # harness: same config, seeds, DGP
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reps", required=True, help="comma-separated replications")
    ap.add_argument("--show", default="", help="file indices to describe (default: "
                    "the most extreme and the most typical file by E2)")
    args = ap.parse_args()

    cfg = H.ExperimentConfig.from_yaml(args.config)
    for rep in (int(x) for x in args.reps.split(",")):
        inspect_rep(H, cfg, rep, args.show)


def inspect_rep(H, cfg, rep: int, show_arg: str) -> None:
    dgp, analyse, truth = H.make_dataset(cfg)
    print(f"\n##### replication {rep}", flush=True)
    real = dgp.sample(cfg.n_org, np.random.default_rng(H.derive_seed(cfg.seed0, rep, 0, "data")))
    gen = H.make_generator(cfg).fit(real, train_seed=H.derive_seed(cfg.seed0, rep, 2, "train"))
    files, Q, U = [], [], []
    for i in range(cfg.m):
        f = gen.generate(cfg.n_syn, gen_seed=H.derive_seed(cfg.seed0, rep, 2, "gen", i))
        files.append(f)
        q, u = analyse(f)
        Q.append(q); U.append(u)
        print(f"file {i}: rejected rows {H._rejected(gen)}, redrawn calls "
              f"{getattr(gen, 'last_n_redrawn_calls', 0)}", flush=True)
    Q, U = np.array(Q), np.array(U)

    # verify against the stored per-file estimates
    paths = glob.glob(f"results/raw/{cfg.name}__{cfg.hash()}/files_*.csv")
    stored = (pd.concat(pd.read_csv(p) for p in paths) if paths
              else pd.DataFrame(columns=["rep", "step", "file", "estimand", "q"]))
    stored = stored[(stored.rep == rep) & (stored.step == 2)]
    if stored.empty:
        print(f"WARNING: no stored per-file values for rep {rep}; reconstruction unverified")
    names = list(H.ESTIMAND_NAMES)
    diffs = []
    for i in range(cfg.m):
        for j, e in enumerate(names):
            s = stored[(stored.file == i) & (stored.estimand == e)]
            if len(s):
                diffs.append(abs(float(s.q.iloc[0]) - Q[i, j]))
    print(f"\nreconstruction check: {len(diffs)} stored values, "
          f"max |stored q - rebuilt q| = {max(diffs) if diffs else float('nan'):.3g}")

    j2 = names.index("E2")
    z = (Q[:, j2] - np.median(Q[:, j2])) / np.sqrt(U[:, j2])
    print("E2 per file (z vs rep median):", np.round(z, 1))
    show = ([int(s) for s in show_arg.split(",")] if show_arg
            else [int(np.argmax(np.abs(z))), int(np.argmin(np.abs(z)))])
    for i in show:
        describe_file(files[i], real, f"rep {rep}, Step 2, file {i} (z = {z[i]:.1f})")
        files[i].to_csv(f"results/bench/inspect_tabpfn_rep{rep}_file{i}.csv", index=False)
    real.to_csv(f"results/bench/inspect_tabpfn_rep{rep}_real.csv", index=False)


if __name__ == "__main__":
    main()
