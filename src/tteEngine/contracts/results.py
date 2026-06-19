"""Result contracts: TTEResult (#10 output) and ComparisonResult (#11 output).

Minimal + extensible seam between the TTE engine (#10) and the
emulated-vs-observed benchmark (#11). The engine returns a TTEResult; the
benchmark compares it to the trial's reported effect. `extra` lets the engine
attach method-specific fields (balance tables, KM curves, E-values, diagnostics)
without changing the seam.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EffectMeasure(str, Enum):
    RR = "RR"   # risk ratio
    RD = "RD"   # risk difference
    OR = "OR"   # odds ratio
    HR = "HR"   # hazard ratio


class TTEResult(BaseModel):
    nct_id: str
    dataset: str
    method: str = Field(..., description="Estimator, e.g. crude_rr / cox_ph / iptw / aipw.")
    measure: EffectMeasure
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_treated: int = 0
    n_control: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class Agreement(str, Enum):
    CONCORDANT = "concordant"
    DISCORDANT = "discordant"
    INCONCLUSIVE = "inconclusive"


class ComparisonResult(BaseModel):
    nct_id: str
    dataset: str
    emulated: TTEResult
    observed_estimate: float | None = None
    observed_measure: EffectMeasure | None = None
    agreement: Agreement = Agreement.INCONCLUSIVE
    notes: str | None = None
