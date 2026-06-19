# Emulating Randomized Trials at Scale Across Multiple ICU EHR Datasets

**Paper skeleton + RESULTS (#63).** Venue target: a clinical-informatics / methods
journal (e.g. *npj Digital Medicine*, *JAMIA*, *Lancet Digital Health*).

> Numbers in **[brackets]** are auto-populated from the live MIMIC/eICU corpus run
> (`tteEngine.analysis.write_narrative` over `contracts.io.load_comparisons_jsonl`;
> see the generated `RESULTS_NARRATIVE.md`). The structure, methods, and limitations
> below are final; the figures are produced by #60 from the same saved corpus.

---

## Abstract

Target trial emulation (TTE) promises to check, and extend, randomized evidence
using routine data — but doing it for *many* trials across *multiple* datasets has
required hand-coding each trial. We built a **trial-general, multi-dataset** TTE
system that (1) reads any ClinicalTrials.gov trial, (2) derives a dataset-agnostic
extraction plan, (3) materializes the emulated cohort from eICU-CRD, MIMIC-IV (and,
pending data access, MGB) normalized to a **common event-stream format**, (4) runs
an estimand-aware causal analysis, and (5) compares the emulated effect to the
trial's *reported* result. Across **[N]** emulable trials × 2 datasets, the
emulation reproduced the trial's direction in **[X]%** (95% CI **[..–..]**);
calibration slope **[s]** (CI coverage **[c]%**); between-trial heterogeneity
I² = **[i2]%**. In the **sepsis** subgroup, concordance was **[Xs]%**. Concordance
was highest for **[top driver]** and lowest for **[bottom driver]**, localizing
where EHR-based emulation is trustworthy.

## 1. Introduction

- TTE framework (Hernán & Robins) as the bridge from RCT to observational data.
- The gap: prior work emulates *one* trial in *one* dataset, hand-coded. The
  open question — *how faithfully can emulation reproduce trials at scale, and
  what predicts success/failure?* — needs a general, multi-dataset pipeline.
- Contribution: an automated trial→emulation→comparison system + a corpus-scale
  concordance/calibration analysis, with sepsis as the deep-dive.

## 2. Methods (summary)

- **Trial ingestion (#1/#2):** ClinicalTrials.gov → typed `TargetTrialSpec`
  (eligibility, treatment/comparator arms, outcomes, time-zero, estimand).
- **Extraction intelligence (#3):** spec → dataset-agnostic `ExtractionPlan`
  (concepts + roles), resolved per dataset via a controlled vocabulary (#5).
- **Common format (#4):** a 5-column long event-stream; per-dataset adapters
  (#6/#7/#8) emit it; cohort views materialized on demand.
- **Cohort (#9):** eligibility, treatment-strategy arm assignment, **landmark
  time-zero** (immortal-time-bias safe).
- **Estimators (#10):** propensity-score matching, IPTW (stabilized, trimmed),
  Cox PH, AIPW doubly-robust; estimand variants (ITT / per-protocol, #38) and
  prespecified subgroups.
- **Benchmark (#11):** emulated effect vs the trial's reported effect →
  concordant / discordant / inconclusive.
- **Meta-analysis (#64):** concordance rate + Wilson CI; DerSimonian–Laird
  random-effects pooling with I²/τ²; sepsis subgroup.
- **Calibration (#41):** emulated-vs-observed slope/intercept, CI coverage.
- **Robustness (#37/#39):** E-values, negative-control outcomes, and
  positive/negative **control trials** for whole-pipeline calibration.

## 3. Results

**Table 1 — Corpus.** Trials fetched / results-posted / emulable, by dataset
(sepsis-prioritized). [counts]

**Table 2 — Concordance.** Overall **[X]%** (95% CI [..–..], n=[k]); sepsis
**[Xs]%**; per-dataset (MIMIC-IV, eICU-CRD). [from #64]

**Table 3 — Calibration.** Slope **[s]**, intercept **[b]**, CI coverage
**[c]%**, RMSE **[r]**. [from #41]

**Flagship.** [The headline sentence: emulation reproduces real RCT direction in
X% of trials, well-calibrated, with the sepsis finding.]

**Drivers (#61).** Ranked features predicting concordance (endpoint type, effect
magnitude, dataset, follow-up, measurability). [from concordance_drivers]

*Figures (#60): forest plot (per-trial + pooled diamond), calibration plot,
representative KM.*

## 4. Limitations

- **Protocol-vs-data gaps (#33/#34):** some eligibility/outcomes are unmeasurable
  or proxy-only in ICU EHR; quantified per trial × dataset.
- **Residual confounding:** observational; bounded by E-values + negative controls,
  not eliminated.
- **Population/scope:** ICU-only cohorts; reported-effect parsing limited to
  count outcomes; follow-up windows differ from the source trials.
- **MGB gated:** the third dataset awaits human-verified data access (extractor
  built, not yet run).

## 5. Conclusion

A general TTE system can emulate many trials across datasets and *quantify its own
fidelity* — concordance, calibration, and the conditions under which EHR-based
emulation can, and cannot, stand in for a randomized trial.
