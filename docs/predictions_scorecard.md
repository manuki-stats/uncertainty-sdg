# Pre-registered predictions and their verdicts

All predictions recorded before the corresponding outcomes were read, with their
verdicts. Coverage refers to $T_{\mathrm{alt}}$ (the naive interval in Step 1) unless
stated otherwise; κ̂ = b_m / ū_m, with subscripts indexing the step.

**Provenance.**

- *Simulation (1–8).* Recorded in the experimental decision note before the simulation
  runs of 9–15 October 2026, and copied here after the thesis was drafted. They
  were not committed to this repository at the time.
- *ACS (T1–T5, S1–S12).* Pre-registered in `docs/preregistration_acs.md`, committed in
  619e745 (16 October 2026) before any coverage of an ACS cell was read. Amendments
  are appended there: D1 (before unblinding) and N1 (after unblinding, no prediction
  changed).
- *Dequantised TabPFN (P1–P2).* Recorded on 29 October 2026, before any output of the
  exploratory cell `acs_tabpfn_dequant` was read.

ACS verdicts follow the pre-registered rule: a cell is *valid* if the 95% Wilson
interval of its coverage contains 0.95, *under-covers* if the upper bound is below
0.93, *over-covers* if the lower bound exceeds 0.95, and is *inconclusive* otherwise.

## Simulation (8 predictions, all held)

| # | Prediction | Verdict | Outcome |
|---|---|---|---|
| 1 | synthpop, both modes: naive Step 1 under-covers (≈ 74% plug-in, ≈ 62% proper) | held | 72–76% / 56–61% |
| 2 | synthpop, both modes: Steps 2–4 near 95% | held | 93.8–95.1% |
| 3 | synthpop: Step 3 reproduces Step 2 | held | |
| 4 | synthpop, E1: κ̂ ≈ 1 (plug-in), ≈ 3 (proper) in Steps 2–3 | held | 1.00–1.02 / 3.07–3.14 |
| 5 | TVAE under-covers in Steps 1–2 on every estimand | held | 8–54% |
| 6 | CTGAN under-covers badly in Steps 1–2 on every estimand | held | 0–15% |
| 7 | CTGAN under-covers on E3 in every step | held | 0% |
| 8 | CTGAN: T_s under-covers in Steps 3–4 | held | 44–45% on E1, 0% on E2 and E3 |

## ACS: theory predictions (3 held, 1 inconclusive, 1 failed)

| # | Prediction | Verdict | Outcome |
|---|---|---|---|
| T1 | Naive Step 1 under-covers in every cell; synthpop's naive variance ratio within ±10% of 1/(r + κ̂₂) | held | within 8% |
| T2 | synthpop, Step 2, E1: κ̂ in [0.85, 1.15] plug-in and [2.6, 3.6] proper | failed | plug-in 0.98; proper 3.68 (per-variable bootstrap of the proper mode) |
| T3 | Plug-in synthpop: Step 3 reproduces Step 2 | held | κ̂₃ − κ̂₂ = +0.04 / −0.02 / −0.02, within 2 Monte Carlo SE |
| T4 | Against q̂_org, Step 1: plug-in naive interval valid on E1; proper under-covers with variance ratio within ±15% of 1/κ̂₂ | inconclusive | plug-in 92.7% [90.9, 94.2]; proper clause holds (within 1–5%) |
| T5 | Against q̂_org: every Step-2 cell with κ̂ ≤ 1.5 and \|standardised bias\| ≥ 0.5 covers below 80% | held | six cells, 0–19% |

## ACS: transfer bets from the simulation (9 held, 3 failed)

| # | Prediction | Verdict | Outcome |
|---|---|---|---|
| S1 | synthpop: no under-coverage in Steps 2–4, either mode | held | |
| S2 | Proper synthpop: κ̂ above 3 on E2 and E3 in Steps 2–3 | held | 3.72–3.88 |
| S3 | TVAE: Steps 1–2 under-cover on every estimand | held | upper Wilson bounds ≤ 52% |
| S4 | TVAE, Step 3: under-covers on E2 and E3, variance ratio below 0.5 on at least one estimand | held | ρ = 0.22 on E1 |
| S5 | TVAE: Step-4 variance ratio above Step 3 on every estimand | held | 0.85 / 0.68 / 0.68 vs 0.22 / 0.38 / 0.46 |
| S6 | TVAE: E3 under-covers in every step | held | |
| S7 | CTGAN: Steps 1–2 under-cover on every estimand | held | |
| S8 | CTGAN: T_s and T_s(PPD) under-cover in Steps 3–4 on every estimand | failed | valid on E3 (91–95%), with over-wide intervals |
| S9 | CTGAN: Step-3 κ̂ above 10 on every estimand | failed | 2.6 on E2, 2.2 on E3; the bet contradicted its own source (simulation: 5.6 on E3), see note N1 |
| S10 | CTGAN: E3 under-covers in Steps 1–3 | failed | Step 3: 93% [86, 97], valid |
| S11 | TabPFN: Step-2 κ̂ on E1 in [0.8, 1.3] | held | 1.01 |
| S12 | TabPFN: κ̂₃ / κ̂₂ on E1 in [0.67, 1.5] | held | 1.28 |

## Dequantised TabPFN (exploratory, 2 held)

| # | Prediction | Verdict | Outcome |
|---|---|---|---|
| P1 | Step-2 bias of E1 below 2 original-data SE in absolute value (as shipped: −6.6 SE) | held | −0.7 SE |
| P2 | Step-2 variance ratio above 1.3 on E2 and E3 | held | 1.67 and 2.00 |

Sources: `results/summary/` (all coverage, κ̂ and variance ratios), and for the
response diagnostics `results/summary/acs_response_diagnostics.csv`.
