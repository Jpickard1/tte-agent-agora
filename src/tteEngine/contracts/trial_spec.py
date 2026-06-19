"""TargetTrialSpec — the structured protocol parsed from a ctgov trial (#2, probe).

PICO-T + estimand. Produced by ctgov parsing (#2); refined/edited in the
check-&-correct loop (#12); consumed by the extraction-intelligence (#3) which
turns it into an ExtractionPlan. v0 skeleton — probe owns the final field set.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .events import EventType


class Estimand(str, Enum):
    INTENTION_TO_TREAT = "itt"
    PER_PROTOCOL = "per_protocol"
    AS_TREATED = "as_treated"


class Comparator(str, Enum):
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    EQ = "eq"
    IN = "in"
    EXISTS = "exists"


class EligibilityCriterion(BaseModel):
    """One inclusion/exclusion predicate, expressed against the event stream."""

    concept: str = Field(..., description="Concept id or raw name resolved via vocab (#5).")
    event_type: EventType
    comparator: Comparator = Comparator.EXISTS
    value: float | str | list[str] | None = None
    unit: str | None = None
    window_hours: tuple[float, float] | None = Field(
        None, description="Time window relative to time-zero, e.g. (-24, 0)."
    )
    include: bool = Field(True, description="True=inclusion, False=exclusion.")


class Arm(BaseModel):
    name: str
    is_control: bool = False
    intervention_concepts: list[str] = Field(default_factory=list)


class OutcomeSpec(BaseModel):
    name: str
    event_type: EventType = EventType.OUTCOME
    concept: str | None = None
    horizon_hours: float | None = Field(None, description="e.g. 28*24 for 28-day mortality.")
    kind: str = Field("binary", description="binary | time_to_event | continuous")


class TimeZeroRule(BaseModel):
    """Landmark/grace-window time-zero — guards against immortal-time bias."""

    anchor: str = Field("icu_admission", description="Event anchoring t0.")
    grace_window_hours: float = Field(24.0, description="Exposure-assessment window after anchor.")


class TargetTrialSpec(BaseModel):
    nct_id: str
    title: str | None = None
    condition: str | None = None
    eligibility: list[EligibilityCriterion] = Field(default_factory=list)
    arms: list[Arm] = Field(default_factory=list)
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    covariates: list[str] = Field(default_factory=list)
    time_zero: TimeZeroRule = Field(default_factory=TimeZeroRule)
    estimand: Estimand = Estimand.INTENTION_TO_TREAT
