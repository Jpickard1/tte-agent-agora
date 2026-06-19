"""ExtractionPlan — SEAM (a): what to pull from a DB for one trial (#3 -> #6/#7/#8).

The extraction-intelligence (#3, probe) turns a TargetTrialSpec into a
dataset-AGNOSTIC plan of the concepts/events required. Each per-DB adapter
(#6/#7/#8, worker1) consumes the same plan and resolves it against its own raw
schema via the vocab layer (#5), emitting the canonical 5-col stream.

This is the contract that lets the intelligence and the adapters be written
independently. v0 skeleton.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .events import EventType
from .timing import TimingConfig


class ConceptRequest(BaseModel):
    """One concept the adapter must locate + extract for this trial."""

    concept: str = Field(..., description="Cross-DB concept id (vocab #5), or raw name to resolve.")
    event_type: EventType
    role: str = Field(..., description="eligibility | exposure | outcome | covariate")
    require_value: bool = Field(False, description="Numeric value+unit needed (e.g. lab threshold).")


class ExtractionPlan(BaseModel):
    nct_id: str
    dataset: str | None = Field(None, description="Target DB tag, or None for dataset-agnostic plan.")
    concepts: list[ConceptRequest] = Field(default_factory=list)
    cohort_filter_concepts: list[str] = Field(
        default_factory=list, description="Concepts defining the extractable cohort (e.g. sepsis dx)."
    )
    window_hours: tuple[float, float] = Field(
        (-48.0, 24.0), description="Extraction window around the anchor event."
    )
    timing: TimingConfig | None = Field(
        None, description="Shared cross-dataset timing contract (#31); None -> use window_hours."
    )
    notes: str | None = None
