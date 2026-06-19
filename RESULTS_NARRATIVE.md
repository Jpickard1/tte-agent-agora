# TTE Emulation — Results Narrative

> NOTE: numbers below are from a SYNTHETIC illustrative corpus to show the
> narrative shape. Regenerate from the live MIMIC/eICU corpus:
>   write_narrative(concordance_drivers(load_comparisons_jsonl(corpus)), meta=..., calibration=...)


_Auto-generated from the emulated-vs-observed corpus._

## Headline
- Across 60 comparable trial-emulations, the emulation reproduced the real RCT's direction in **73%** (95% CI 61%–83%).
- Calibration slope **0.99** (ideal 1.0), CI coverage **100%** over 60 trials.
- Between-trial heterogeneity I² = **78%**.
- **Sepsis:** Sepsis concordance 80% (n=20) vs non-sepsis 70% (n=40).

## What predicts concordance (ranked drivers)
- **ci_width_log** (continuous, 28% rate spread): < 0.693=100% (n=2); >= 0.693=72% (n=58)
- **dataset** (categorical, 27% rate spread): MIMIC-IV=87% (n=30); eICU=60% (n=30)
- **effect_magnitude** (continuous, 13% rate spread): < 0.366=67% (n=30); >= 0.366=80% (n=30)
- **is_sepsis** (categorical, 10% rate spread): False=70% (n=40); True=80% (n=20)
- **e_value** (continuous, 7% rate spread): < 1.31=70% (n=30); >= 1.31=77% (n=30)

## Where emulation succeeds / fails
- Strata with the highest concordance mark where TTE is trustworthy; the lowest mark failure modes (residual confounding, outcome-proxy mismatch, sparse data).

_Numbers populate from the live MIMIC/eICU corpus run; regenerate via `write_narrative(concordance_drivers(load_comparisons_jsonl(corpus)), ...)`._