"""Appendix tables built from the committed summaries (results/summary).

  B.1  T_f under the oracle: negative-variance rate and coverage after the fallback
  C.1  simulation, Steps 2-4: T_alt coverage with Wilson interval, T_s / T_s(PPD), bias
  C.2  ACS, same layout, with the pre-registered verdict marked on T_alt

    python scripts/make_appendix_tables.py --summary results/summary --out appendix_tables/

Bias is in original-data standard errors: the exact sqrt(U) of the simulation's
data-generating process, and sqrt(mean u_org) on the ACS data (0.0106 / 0.0115 / 0.0164).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIM = [("\\texttt{synthpop} plug-in", "synthpop_plugin__f6031a194e"),
       ("\\texttt{synthpop} proper", "synthpop_proper__2ad55ad86b"),
       ("TVAE", "tvae__7c8fff32f1"),
       ("CTGAN", "ctgan__dfc3687fa3"),
       ("TabPFN", "tabpfn__01685b28f1")]
ACS = [("\\texttt{synthpop} plug-in", "acs_synthpop_plugin__c16f1898b6"),
       ("\\texttt{synthpop} proper", "acs_synthpop_proper__084ab52bf2"),
       ("TVAE", "acs_tvae__e1f9b9ee67"),
       ("CTGAN", "acs_ctgan__5302eb2c45"),
       ("TabPFN", "acs_tabpfn__42033f7c9b"),
       ("TabPFN dequantised", "acs_tabpfn_dequant__d12440235d")]
ORACLE = [("Plug-in", "oracle_plugin__86aa55c467"),
          ("Proper", "oracle_proper__4790a5b577"),
          ("Single-draw", "oracle_single_draw__a0ac933122")]
SE_SIM = {"E1": np.sqrt(8.36 / 5000), "E2": np.sqrt(1 / 3000), "E3": np.sqrt(1 / 3000 + 1 / 2000)}
SE_ACS = {"E1": 0.0106, "E2": 0.0115, "E3": 0.0164}
EST = ["E1", "E2", "E3"]


def pct(x: float, R: int) -> str:
    return f"{100 * x:.0f}" if R < 1000 else f"{100 * x:.1f}"


def verdict(lo: float, hi: float) -> str:
    if lo <= 0.95 <= hi:
        return ""
    if hi < 0.93:
        return "$^{u}$"
    if lo > 0.95:
        return "$^{o}$"
    return "$^{i}$"


def signed(x: float) -> str:
    s = f"{x:+.1f}".replace("-", "$-$")
    return "0.0" if s in ("+0.0", "$-$0.0") else s


def row_block(name: str, s: pd.DataFrame, se: dict, steps, mark: bool) -> list[str]:
    lines = []
    for k, st in enumerate(steps):
        cells = [name if k == 0 else "", str(st)]
        for e in EST:
            g = s[(s.step == st) & (s.estimand == e)].set_index("estimator")
            a = g.loc["T_alt"]
            R = int(a.R)
            v = verdict(a.wilson_lo, a.wilson_hi) if mark else ""
            cells += [f"{pct(a.coverage, R)}{v} [{pct(a.wilson_lo, R)}, {pct(a.wilson_hi, R)}]",
                      f"{pct(g.loc['T_s'].coverage, R)}/{pct(g.loc['T_s_ppd'].coverage, R)}",
                      signed(a.bias / se[e])]
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def results_table(cells, se, mark: bool, caption: str, label: str, root: Path) -> str:
    head = ["\\begin{table}[htbp]", "  \\centering", "  \\footnotesize",
            "  \\setlength{\\tabcolsep}{3pt}",
            "  \\resizebox{\\textwidth}{!}{%",
            "  \\begin{tabular}{ll" + "lll" * 3 + "}", "    \\toprule",
            "    & & \\multicolumn{3}{c}{E1} & \\multicolumn{3}{c}{E2} & \\multicolumn{3}{c}{E3} \\\\",
            "    \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
            "    Generator & Step" + " & $T_{\\mathrm{alt}}$ [Wilson] & $T_s$/PPD & Bias" * 3 + " \\\\",
            "    \\midrule"]
    body = []
    for i, (name, cell) in enumerate(cells):
        s = pd.read_csv(root / f"{cell}.csv")
        steps = sorted(set(s.step) - {1})
        if i:
            body.append("    \\addlinespace")
        body += ["    " + l for l in row_block(name, s, se, steps, mark)]
    tail = ["    \\bottomrule", "  \\end{tabular}}",
            f"  \\caption{{{caption}}}", f"  \\label{{{label}}}", "\\end{table}"]
    return "\n".join(head + body + tail) + "\n"


def tf_table(root: Path) -> str:
    lines = ["\\begin{table}[htbp]", "  \\centering", "  \\small",
             "  \\begin{tabular}{llll}", "    \\toprule",
             "    Oracle mode & Step & $T_f < 0$ (\\%) & Coverage (\\%) \\\\", "    \\midrule"]
    for name, cell in ORACLE:
        s = pd.read_csv(root / f"{cell}.csv")
        t = s[s.estimator == "T_f"]
        for st in (2, 3, 4):
            g = t[t.step == st]
            neg = f"{100 * g.corrected_rate.min():.1f}--{100 * g.corrected_rate.max():.1f}"
            cov = f"{100 * g.coverage.min():.1f}--{100 * g.coverage.max():.1f}"
            lines.append(f"    {name if st == 2 else ''} & {st} & {neg} & {cov} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}",
              "  \\caption{Rubin's $T_f$ under the oracle ($R = 1{,}000$, $r = 2$, $m = 10$), ranges over E1--E3. "
              "When $T_f$ is negative the interval falls back to $r\\bar u_m$.}",
              "  \\label{tab:oracle-tf}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary")
    ap.add_argument("--out", default="appendix_tables")
    a = ap.parse_args()
    root, out = Path(a.summary), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_oracle_tf.tex").write_text(tf_table(root))
    (out / "tab_sim_full.tex").write_text(results_table(
        SIM, SE_SIM, False,
        "Simulation, Steps~2--4: coverage (\\%) of $T_{\\mathrm{alt}}$ with its 95\\% Wilson interval, "
        "coverage of $T_s$ and $T_{s(\\mathrm{PPD})}$, and the bias of $\\bar q_m$ in original-data standard "
        "errors $\\sqrt{U}$. $R = 1{,}000$ except for CTGAN ($R = 100$).",
        "tab:sim-full", root))
    (out / "tab_acs_full.tex").write_text(results_table(
        ACS, SE_ACS, True,
        "ACS, Steps~2--4 against the population values, laid out as Table~\\ref{tab:sim-full}. "
        "Superscripts give the pre-registered verdict when it is not \\emph{valid}: "
        "$u$ under-covers, $o$ over-covers, $i$ inconclusive. $R = 1{,}000$ except for CTGAN and "
        "the dequantised TabPFN cell ($R = 100$; Step~2 only for the latter).",
        "tab:acs-full", root))
    print(f"wrote {out}/tab_oracle_tf.tex, tab_sim_full.tex, tab_acs_full.tex")


if __name__ == "__main__":
    main()
