"""TimingConfig — ONE timing contract across eICU / MIMIC / MGB (#31, worker1).

The three datasets keep time differently: MIMIC-IV carries wall-clock timestamps
(sub-minute), eICU-CRD carries OFFSETS in minutes from unit admission (no
wall-clock, by design), MGB carries Snowflake timestamps. Left implicit, that
makes estimands silently incomparable — a "-48h..24h window" or a "minute"
resolution means something slightly different in each.

`TimingConfig` makes the timing EXPLICIT and shared: one config defines the
reference clock, the extraction window, lookback/washout windows, the
treatment-assignment grace window, and a COMMON harmonized timestamp precision.
The same config drives every adapter (via tteEngine.timing.effective_window +
harmonize_timestamps) so the windows + precision are identical across datasets.

Back-compat: ExtractionPlan.timing defaults to None -> adapters fall back to the
legacy plan.window_hours and skip harmonization, so existing behavior is unchanged.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class ClockReference(str, Enum):
    """Which admission anchors t0. ICU admission is the comparable default across
    ICU datasets; hospital admission is available where the DB records it."""
    ICU_ADMISSION = "icu_admission"
    HOSPITAL_ADMISSION = "hospital_admission"


class TimePrecision(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class TimingConfig(BaseModel):
    """The single timing contract driving all adapters so estimands are
    comparable across datasets. All windows are HOURS relative to time-zero."""

    clock: ClockReference = ClockReference.ICU_ADMISSION
    extraction_window_hours: tuple[float, float] = Field(
        (-48.0, 24.0), description="Data-pull window around t0 (lo<=hi).")
    lookback_hours: float = Field(48.0, ge=0.0, description="Covariate/eligibility lookback before t0.")
    washout_hours: float = Field(0.0, ge=0.0, description="No-prior-exposure window before t0.")
    grace_window_hours: float = Field(24.0, ge=0.0, description="Post-t0 grace for treatment assignment.")
    precision: TimePrecision = Field(
        TimePrecision.MINUTE, description="Common precision all datasets are floored to.")

    @model_validator(mode="after")
    def _check_window(self) -> "TimingConfig":
        lo, hi = self.extraction_window_hours
        if lo > hi:
            raise ValueError(f"extraction_window_hours lo({lo}) must be <= hi({hi})")
        return self


#: each DB's NATIVE timestamp precision — documents what we harmonize FROM (and
#: lets a validator warn if a config asks for finer precision than a DB supports).
DATASET_NATIVE_PRECISION: dict[str, TimePrecision] = {
    "MIMIC-IV": TimePrecision.SECOND,   # charttime is sub-minute
    "eICU-CRD": TimePrecision.MINUTE,   # times are integer-minute offsets
    "MGB": TimePrecision.SECOND,        # Snowflake timestamps (gated)
}


__all__ = ["TimingConfig", "ClockReference", "TimePrecision", "DATASET_NATIVE_PRECISION"]
