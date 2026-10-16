# Real-data study: pre-registered analysis plan and predictions

ACS 2024 pseudo-population · written 16 Oct 2026, committed before any coverage of an `acs_*` cell was summarised or read. The commit is the timestamp; nothing in sections 1–4 changes after unblinding. Deviations are appended in section 5, dated, without editing what precedes it.

State at commit. Pilot submitted (synthpop both modes reps 0–9, TVAE 0–9, CTGAN 0–3, TabPFN 0–7). Its output is read only for run times, logs, failures and TabPFN redraw counts until this file is pushed.

## 1. Fixed in advance

**Data and truth.** National pseudo-population, N = 717,219 (extracts/acs2024_us/manifest.json); tuning sample of 5,000 reserved before the population was formed. Analysis model M2: log wage on a, female, a × female, a², education dummies (reference: high school), a = (age − 45)/10. Truth: the population's M2 values (population_truth.json): E1 mean log wage 11.0389, E2 coefficient of a 0.0907, E3 coefficient of a × female −0.0365. Design-based check at n_org = 5000 (2,000 draws): complete-data intervals cover 0.950–0.955, variance ratios 1.017–1.018.

**Cells** (config hash): acs_synthpop_proper 084ab52bf2, acs_synthpop_plugin c16f1898b6, acs_tabpfn 42033f7c9b, acs_tvae e1f9b9ee67 (tuned: t19), acs_ctgan 5302eb2c45 (tuned: t11). Common to all: n_org 5000, r 2, m = K = 10, Steps 1–4, common random numbers (one D_org per replication, shared by every cell).

**Targets.** Primary: the population value, D_org redrawn each replication (f = 0.70%), full combining rules. Secondary: each replication's complete-data estimate q_org, sampling term dropped, rebuilt from the same runs (`summarise --target sample`). This q_org target averages over D_org draws; a fixed-D_org version, if required on 19 Oct, would be a separate run to which T4 and T5 apply unchanged.

**Estimators.** T_alt primary; T_s and T_s(PPD) secondary; the naive interval in Step 1. t intervals with the estimators' degrees of freedom; α = 0.05.

**Replications.** synthpop (both modes): R = 100, raised to 1000 by the simulation's rule (the Wilson upper bound of T_alt coverage is ≥ 0.93 in any Step 2–4 cell). TVAE, CTGAN, TabPFN: R = 100, no escalation (proposal to be confirmed on 19 Oct; if escalation is required, the simulation's rule applies to every cell).

**Verdicts.** At the final R, each cell (generator × step × estimand × estimator × target) is *valid* if the Wilson 95% interval of its coverage contains 0.95, *under-covers* if the interval's upper bound is below 0.93, *over-covers* if its lower bound is above 0.95, and *inconclusive* otherwise. Standardised bias = mean(q̄ − Q) / √(mean u_org), with u_org the complete-data variance stored in `orig_*`. Every prediction below is reported as held, failed or inconclusive.

## 2. Predictions from theory

These follow from the combining rules and the release designs. A failure points to an implementation error or a violated assumption, and is investigated before anything is interpreted.

- **T1.** Population target, Step 1: the naive interval under-covers in every cell. For synthpop (both modes), its variance ratio equals 1/(r + κ̂₂) within ±10%, κ̂₂ being the Step-2 κ̂ of the same estimand (assumes ū ≈ u/r and negligible bias).
- **T2.** synthpop, Step 2, E1: κ̂ in [0.85, 1.15] for plug-in and in [2.6, 3.6] for proper (theory: 1 and 1 + r = 3).
- **T3.** synthpop plug-in: Step 3 reproduces Step 2 (κ̂₃ within MC error of κ̂₂ on every estimand), because CART's fit does not depend on the training seed.
- **T4.** q_org target, Step 1: synthpop plug-in's naive interval is valid on E1; synthpop proper's under-covers on every estimand, with variance ratio 1/κ̂₂ within ±15%.
- **T5.** Bias magnification under q_org. In every Step-2 cell with κ̂ ≤ 1.5 and standardised bias of absolute value ≥ 0.5, T_alt coverage under the q_org target is below 0.80 (its standard error, √(b/m) ≈ √(κ̂/(r·m)) original standard errors, is 0.27 of one at κ̂ = 1.5).

## 3. Predictions extrapolated from the simulation and the tuning

These are empirical bets on transfer from the simulation. A failure is a finding, not a bug.

synthpop
- **S1.** Population target, T_alt: no under-coverage in Steps 2–4 on any estimand, in either mode (simulation: 0.938–0.959).
- **S2.** Proper: κ̂ above 3 on E2 and E3 in Steps 2–3 (simulation: 3.8–4.6).

TVAE (tuned; stage-2 pMSE ratio 23.4)
- **S3.** Steps 1–2 under-cover on every estimand (simulation: 0.08–0.54).
- **S4.** Step 3 (seed refits) under-covers on E2 and E3, with T_alt variance ratio below 0.5 on at least one estimand.
- **S5.** Step 4 (bootstrap refits) has a higher T_alt variance ratio than Step 3 on every estimand.
- **S6.** E3 under-covers in every step.

CTGAN (tuned; stage-2 pMSE ratio 6.67)
- **S7.** Steps 1–2 under-cover on every estimand.
- **S8.** T_s and T_s(PPD) under-cover in Steps 3–4 on every estimand.
- **S9.** Step-3 κ̂ exceeds 10 on every estimand.
- **S10.** E3 under-covers in Steps 1–3.

TabPFN (defaults)
- **S11.** Step-2 κ̂ on E1 in [0.8, 1.3].
- **S12.** Step 3 ≈ Step 2: κ̂₃/κ̂₂ on E1 in [0.67, 1.5], because a training seed changes only TabPFN's preprocessing.

## 4. No prediction

Stated so that none of these can be presented afterwards as anticipated:
- TabPFN coverage in Steps 2–4, and whether its far-bucket draws recur on these data.
- CTGAN in Step 4 on any estimand; TVAE in Step 4 on E1 and E2.
- Whether tuning on the real data changes the simulation's DGE pattern (CTGAN's valid E1 in Step 3).
- Any cell under the q_org target in Steps 2–4 beyond T5.

Reported regardless: run times, TabPFN redraw counts, fit failures, and the escalation decisions.

## 5. Deviations after commit

(none)
