"""Pipeline calibration via positive/negative control trials (#39, probe).

A whole-system check: take trials whose effect direction is already KNOWN, run
each through the emulation spine, and verify the pipeline recovers the expected
sign (positive controls) or null (negative controls). High recovery on positives
= the pipeline has validity/power; high specificity on negatives (correctly null)
= it isn't manufacturing spurious effects. Extends the #37 negative-control idea
from outcomes to whole trials.

Standalone: `calibrate` takes an injected `emulate(ControlTrial) -> TTEResult`
callable, so it runs against the real Pipeline in production and a stub in tests —
it does not itself depend on the corpus (#36) or real data.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

from pydantic import BaseModel, Field

from ..contracts.results import EffectMeasure, TTEResult


class ExpectedEffect(str, Enum):
    BENEFIT = "benefit"   # treatment lowers the (harmful) outcome: estimate < null
    HARM = "harm"         # estimate > null
    NULL = "null"         # no effect: CI spans the null


class ControlTrial(BaseModel):
    nct_id: str
    label: str
    expected: ExpectedEffect
    kind: str = Field(..., description="'positive' (known non-null) | 'negative' (known null).")
    treatment_hint: str = ""
    note: str = ""


# Illustrative seed suite — REPLACE/EXPAND with the clinically-curated authoritative
# list. The calibration logic below is the deliverable; these are starter examples.
CONTROL_TRIALS: list[ControlTrial] = [
    ControlTrial(nct_id="NCT00147004", label="Low-dose hydrocortisone in septic shock",
                 expected=ExpectedEffect.BENEFIT, kind="positive", treatment_hint="hydrocortisone",
                 note="Positive control: corticosteroid mortality signal in septic shock."),
    ControlTrial(nct_id="NCT00510835", label="Intensive vs conventional glucose control",
                 expected=ExpectedEffect.HARM, kind="positive", treatment_hint="intensive insulin",
                 note="Positive control: intensive glucose control increased mortality (NICE-SUGAR)."),
    ControlTrial(nct_id="NCT00000000", label="Vitamin/placebo-class null exemplar",
                 expected=ExpectedEffect.NULL, kind="negative", treatment_hint="",
                 note="Negative control: a treatment with no expected mortality effect."),
]


def observed_direction(tte: TTEResult) -> ExpectedEffect:
    """Classify an emulated estimate as benefit / harm / null (CI-based)."""
    null = 0.0 if tte.measure == EffectMeasure.RD else 1.0
    if tte.ci_low is not None and tte.ci_high is not None and tte.ci_low <= null <= tte.ci_high:
        return ExpectedEffect.NULL
    if tte.estimate is None:
        return ExpectedEffect.NULL
    return ExpectedEffect.HARM if tte.estimate > null else ExpectedEffect.BENEFIT


class CalibrationResult(BaseModel):
    nct_id: str
    label: str
    kind: str
    expected: ExpectedEffect
    observed: ExpectedEffect
    recovered: bool        # observed direction matches the known expectation
    note: str = ""


class CalibrationReport(BaseModel):
    results: list[CalibrationResult] = Field(default_factory=list)
    n: int = 0
    n_recovered: int = 0
    positive_recovery_rate: float | None = None   # positives whose sign was recovered
    negative_specificity: float | None = None     # negatives correctly returned null
    passed: bool = False
    note: str = ""


def calibrate(control_trials, emulate: Callable[[ControlTrial], TTEResult],
              *, min_positive_recovery: float = 0.7, min_negative_specificity: float = 0.7
              ) -> CalibrationReport:
    """Emulate each control trial and score expected-vs-observed agreement.

    `emulate(ct) -> TTEResult` is injected (the real Pipeline, or a stub). passed
    requires recovery on positives + specificity on negatives to clear thresholds."""
    results: list[CalibrationResult] = []
    for ct in control_trials:
        obs = observed_direction(emulate(ct))
        results.append(CalibrationResult(
            nct_id=ct.nct_id, label=ct.label, kind=ct.kind, expected=ct.expected,
            observed=obs, recovered=(obs == ct.expected),
            note=("recovered expected direction" if obs == ct.expected
                  else f"expected {ct.expected.value}, emulated {obs.value}"),
        ))

    pos = [r for r in results if r.kind == "positive"]
    neg = [r for r in results if r.kind == "negative"]
    pos_rate = (sum(r.recovered for r in pos) / len(pos)) if pos else None
    neg_spec = (sum(r.recovered for r in neg) / len(neg)) if neg else None
    passed = ((pos_rate is None or pos_rate >= min_positive_recovery)
              and (neg_spec is None or neg_spec >= min_negative_specificity)
              and bool(results))
    return CalibrationReport(
        results=results, n=len(results), n_recovered=sum(r.recovered for r in results),
        positive_recovery_rate=pos_rate, negative_specificity=neg_spec, passed=passed,
        note=(f"{sum(r.recovered for r in results)}/{len(results)} controls recovered "
              f"(positive recovery {pos_rate if pos_rate is None else round(pos_rate,2)}, "
              f"negative specificity {neg_spec if neg_spec is None else round(neg_spec,2)})."),
    )
