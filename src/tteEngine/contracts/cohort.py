"""CohortResult — SEAM (c): cohort builder (#9) -> TTE engine (#10).

The cohort builder (probe, #9) reads the canonical 5-col stream + a
TargetTrialSpec, applies eligibility/time-zero/arm assignment, and materializes
an analysis-ready WIDE feature view. CohortResult is what the TTE engine (#10)
consumes. The wide feature table is a deterministic VIEW over the canonical long
stream (probe's must-have #3), never a separate source of truth.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ArmAssignment(BaseModel):
    name: str
    is_control: bool = False
    trajectory_ids: list[int] = Field(default_factory=list)


class CohortDiagnostics(BaseModel):
    """Time-zero + immortal-time + attrition diagnostics (#30). Makes the
    landmark explicit and records every exclusion (no silent attrition)."""

    n_screened: int = 0
    n_eligible: int = 0
    n_excluded_immortal: int = 0
    n_enrolled: int = 0
    anchor: str = ""
    grace_window_hours: float = 0.0
    landmark_hours: float = 0.0
    arm_sizes: dict[str, int] = Field(default_factory=dict)
    leakage_warnings: list[str] = Field(default_factory=list)


class CohortResult(BaseModel):
    nct_id: str
    dataset: str
    arms: list[ArmAssignment] = Field(default_factory=list)
    index_times: dict[int, datetime] = Field(
        default_factory=dict, description="trajectory_id -> time-zero (landmark)."
    )
    feature_table_ref: str | None = Field(
        None, description="Path to the materialized wide analysis frame (parquet)."
    )
    feature_columns: list[str] = Field(default_factory=list)
    n_total: int = 0
    diagnostics: "CohortDiagnostics | None" = None

    def n_by_arm(self) -> dict[str, int]:
        return {a.name: len(a.trajectory_ids) for a in self.arms}
