"""Sensitivity / robustness analyses (#37, probe / lane:analysis).

For unmeasured-confounding robustness, every emulation can report:
- an E-VALUE (already computed by the engine; surfaced + interpreted here) — how
  strong an unmeasured confounder would need to be to explain away the estimate;
- NEGATIVE-CONTROL outcome checks — re-run the same design on outcome(s) the
  treatment should NOT affect; a non-null effect there signals residual
  confounding / bias (the design is "miscalibrated").

`sensitivity_report` combines both into one typed verdict, satisfying #37's
acceptance (E-value + >=1 negative-control check per emulation). Builds on the
ported engine via run_tte; the analysis extra loads lazily through it.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

from ..contracts.results import EffectMeasure, TTEResult
from .runner import run_tte


def _null_value(measure: EffectMeasure) -> float:
    return 0.0 if measure == EffectMeasure.RD else 1.0


def _finite(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _spans_null(r: TTEResult) -> bool:
    """True if the CI covers the null value (i.e. no significant effect)."""
    null = _null_value(r.measure)
    return r.ci_low is not None and r.ci_high is not None and r.ci_low <= null <= r.ci_high


def evalue_note(point: float | None) -> str | None:
    if point is None:
        return None
    return (f"An unmeasured confounder would need to be associated with both treatment and "
            f"outcome by a risk ratio of {point:.2f}-fold each (beyond the measured covariates) "
            f"to fully explain away the observed estimate.")


class NegativeControlResult(BaseModel):
    """A negative-control outcome re-analysis. `flagged` = a non-null effect on an
    outcome the treatment should not affect → evidence of residual bias."""

    outcome: str
    measure: str
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    flagged: bool = False


class SensitivityReport(BaseModel):
    """E-value + negative-control robustness for one emulation (#37)."""

    e_value_point: float | None = None
    e_value_ci: float | None = None
    e_value_note: str | None = None
    negative_controls: list[NegativeControlResult] = Field(default_factory=list)
    n_controls: int = 0
    n_controls_flagged: int = 0
    passed: bool = False
    note: str = ""


def negative_control_check(frame, *, negative_outcome_cols, covariates,
                           treatment_col: str = "T", **run_tte_kwargs) -> list[NegativeControlResult]:
    """Re-run run_tte on each negative-control outcome. A control is `flagged` when
    its CI does NOT span the null (a spurious effect on something that should be null)."""
    results: list[NegativeControlResult] = []
    for col in negative_outcome_cols:
        r = run_tte(frame, outcome_col=col, covariates=list(covariates),
                    treatment_col=treatment_col, **run_tte_kwargs)
        results.append(NegativeControlResult(
            outcome=col, measure=r.measure.value, estimate=_finite(r.estimate),
            ci_low=_finite(r.ci_low), ci_high=_finite(r.ci_high),
            flagged=(r.extra.get("ok", True) and not _spans_null(r)),
        ))
    return results


def sensitivity_report(primary: TTEResult, frame, *, negative_outcome_cols=(),
                       covariates, treatment_col: str = "T", **run_tte_kwargs) -> SensitivityReport:
    """E-value (from the primary emulation) + negative-control checks -> one verdict.
    `passed` = an E-value is present AND no negative control is flagged."""
    ev = _finite(primary.extra.get("e_value_point"))
    ev_ci = _finite(primary.extra.get("e_value_ci"))
    controls = negative_control_check(
        frame, negative_outcome_cols=negative_outcome_cols, covariates=covariates,
        treatment_col=treatment_col, **run_tte_kwargs)
    n_flagged = sum(c.flagged for c in controls)
    passed = ev is not None and n_flagged == 0
    if not controls:
        note = "No negative-control outcomes supplied — provide >=1 for a bias check."
    elif n_flagged == 0:
        note = (f"Robust: no spurious effect on {len(controls)} negative control(s); "
                f"E-value {ev:.2f}." if ev is not None else
                f"No negative control flagged, but no E-value (effect not significant).")
    else:
        note = (f"CAUTION: {n_flagged}/{len(controls)} negative control(s) show a non-null "
                f"effect — residual confounding likely; interpret the primary estimate cautiously.")
    return SensitivityReport(
        e_value_point=ev, e_value_ci=ev_ci, e_value_note=evalue_note(ev),
        negative_controls=controls, n_controls=len(controls),
        n_controls_flagged=n_flagged, passed=passed, note=note,
    )
