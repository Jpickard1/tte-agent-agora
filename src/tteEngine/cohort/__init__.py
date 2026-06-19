"""cohort: build the emulated-trial cohort from the canonical 5-col stream (#9).

Applies eligibility, treatment-strategy arm assignment, LANDMARK time-zero
(immortal-time safe), covariates + outcomes -> an analysis-ready wide view, and
emits contracts.CohortResult (the #9->#10 seam consumed by the TTE engine).
Reuses emulaTTE cohort_builder's landmark pattern.
"""

from .builder import build_analysis_frame, build_cohort

__all__ = ["build_cohort", "build_analysis_frame"]
