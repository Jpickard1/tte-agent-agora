"""ctgov: trial reader + cache, NCT -> TargetTrialSpec, and the trial ->
ExtractionPlan 'intelligence' (#1/#2/#3, owner: probe).

Emits contracts.TargetTrialSpec and contracts.ExtractionPlan. Reuses trialsim
fetch_ctgov.py + emulaTTE schemas.
"""

from .reader import (
    fetch_batch,
    fetch_study,
    nct_id_of,
    reported_outcome_measures,
)

__all__ = [
    "fetch_study",
    "fetch_batch",
    "nct_id_of",
    "reported_outcome_measures",
]
