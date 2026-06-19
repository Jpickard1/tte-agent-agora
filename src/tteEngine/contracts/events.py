"""The canonical common EHR format: a 5-column LONG event-stream (#4).

This is the single source of truth every per-DB adapter (#6/#7/#8) normalizes
into and every downstream consumer (cohort builder #9, engine #10) reads from.
It is the EHR-DE format (already emitted by EHR-DE/MIMIC-IV/extraction_v1.py and
the MGB Snowflake pipeline), promoted here to the project-wide contract.

On-disk canonical = parquet with exactly these 5 columns and dtypes:

    TRAJECTORY_ID  int64                 one hospital admission (>=1 ICU stay)
    TIMESTAMP      datetime64[ns, UTC]   event time; sub-day, orderable (immortal-time safe)
    EVENT_TYPE     category/str          one of EventType below
    EVENT_NAME     str                   source field identifier (raw, pre-normalization)
    EVENT_VALUE    str                   measured value or JSON metadata

The vocab layer (#5) does NOT mutate these 5 columns; it produces a SIDECAR
normalized view (concept_id / value_num / unit / value_text) keyed by row, so
structure (this file) and semantics (#5) stay decoupled. `NormalizedEvent`
documents that sidecar shape.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Canonical event domains — the controlled vocabulary for EVENT_TYPE, as
    actually emitted by EHR-DE/MIMIC-IV/extraction_v1.py + the MGB Snowflake
    pipeline. Extensible: add members here; adapters MUST NOT emit free strings.
    """

    MEASUREMENT = "measu"   # vitals / charted numeric measurements
    LAB = "lab"             # laboratory results
    MEDICATION = "medic"    # drug administrations / prescriptions
    DIAGNOSIS = "diagn"     # ICD diagnoses
    PROCEDURE = "proce"     # procedures
    LOCATION = "locat"      # ward / ICU location & transfers
    DEMOGRAPHIC = "demog"   # age / sex / etc.
    OUTCOME = "outco"       # mortality / discharge / trial endpoints
    MICRO = "micro"         # microbiology (organism / antibiotic / interpretation)
    EMAR = "emar"           # electronic medication administration records
    DRG = "drg"             # diagnosis-related groups
    ORDER = "order"         # orders


#: TRAJECTORY_ID reserved for data-dictionary rows (MGB/EHR-DE convention).
DICTIONARY_TRAJECTORY_ID: int = 0


#: Column order/names of the canonical parquet. Adapters MUST emit exactly these.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "TRAJECTORY_ID",
    "TIMESTAMP",
    "EVENT_TYPE",
    "EVENT_NAME",
    "EVENT_VALUE",
)

#: Pandas dtypes for the canonical parquet (validated by common_format.validate_canonical).
CANONICAL_DTYPES: dict[str, str] = {
    "TRAJECTORY_ID": "int64",
    "TIMESTAMP": "datetime64[ns, UTC]",
    "EVENT_TYPE": "object",
    "EVENT_NAME": "object",
    "EVENT_VALUE": "object",
}


class Event(BaseModel):
    """Row view of one canonical event (1:1 with a parquet row)."""

    trajectory_id: int = Field(..., description="Hospital admission id; 0 reserved for dictionary rows.")
    timestamp: datetime = Field(..., description="UTC, sub-day precision, orderable.")
    event_type: EventType
    event_name: str = Field(..., description="Raw source field name, pre-normalization.")
    event_value: str = Field(..., description="Value as string, or JSON for structured metadata.")


class NormalizedEvent(BaseModel):
    """Sidecar produced by the vocab layer (#5). Keyed to an Event; never replaces it.

    Carries the typed value + unit the analysis layer (#10) requires for
    threshold-based eligibility/outcomes (probe's must-have #1).
    """

    trajectory_id: int
    timestamp: datetime
    event_type: EventType
    concept_id: str | None = Field(None, description="Cross-DB concept (ICD/RxNorm/LOINC/SNOMED).")
    value_num: float | None = Field(None, description="Numeric value when measurable.")
    unit: str | None = Field(None, description="Normalized unit (e.g. mg/dL).")
    value_text: str | None = Field(None, description="Categorical/text value when not numeric.")
