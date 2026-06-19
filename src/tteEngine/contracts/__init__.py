"""Typed boundaries every lane codes to (#4, tte1).

The pipeline is a chain of typed contracts; each seam lets two lanes be built
independently:

    ctgov trial
      -> TargetTrialSpec        (#2 probe)
      -> ExtractionPlan         SEAM (a): #3 probe emits -> #6/#7/#8 worker1 consume
      -> Event stream (5-col)   SEAM (b): adapters emit  -> #9 cohort consumes   [events.py]
      -> CohortResult           SEAM (c): #9 probe emits -> #10 engine consumes
      -> TTEResult              (#10 probe) -> benchmark vs observed (#11)
"""

from .cohort import ArmAssignment, CohortResult
from .events import (
    CANONICAL_COLUMNS,
    CANONICAL_DTYPES,
    Event,
    EventType,
    NormalizedEvent,
)
from .extraction_plan import ConceptRequest, ExtractionPlan
from .trial_spec import (
    Arm,
    Comparator,
    EligibilityCriterion,
    Estimand,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "CANONICAL_DTYPES",
    "Event",
    "EventType",
    "NormalizedEvent",
    "ConceptRequest",
    "ExtractionPlan",
    "TargetTrialSpec",
    "EligibilityCriterion",
    "Arm",
    "OutcomeSpec",
    "TimeZeroRule",
    "Estimand",
    "Comparator",
    "ArmAssignment",
    "CohortResult",
]
