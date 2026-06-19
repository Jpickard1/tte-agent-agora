"""Engine provider: plug the real TTE engine (#10) into the Pipeline (#12).

The bridge between the #9 cohort seam and probe's #10 estimator. It builds the
analysis-ready frame from a CohortResult, adds the 0/1 treatment indicator,
picks the outcome + covariates, calls ``run_tte``, and returns a
``contracts.results.TTEResult`` (the single public seam).

Kept OUT of orchestration/__init__ and importing analysis LAZILY, so
``import tteEngine.orchestration`` stays light (the crude baseline runs with no
analysis extra); the heavy estimators load only when you actually use this
provider. Override the Pipeline 'tte' provider with `make_engine_provider(...)`
to run the real (e.g. IPTW/PSM) estimate instead of the crude baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from tteEngine.contracts.results import EffectMeasure, TTEResult

if TYPE_CHECKING:
    from tteEngine.common_format import FeatureSpec

_MEASURE = {
    "Hazard Ratio": EffectMeasure.HR,
    "Odds Ratio": EffectMeasure.OR,
    "Risk Difference": EffectMeasure.RD,
    "Risk Ratio": EffectMeasure.RR,
    "Relative Risk": EffectMeasure.RR,
}


def _to_contract(r, spec, cohort, adjustment: str) -> TTEResult:
    """Normalize the engine's return into contracts.TTEResult.

    Pass-through once #10/#11 return the contract directly; otherwise map the
    transitional analysis.TTEResult (duck-typed on `point_estimate`).
    """
    if isinstance(r, TTEResult):
        # run_tte sees only the analysis frame, so it cannot know the trial /
        # dataset identity — stamp them from spec/cohort if left blank.
        patch = {}
        if not r.nct_id:
            patch["nct_id"] = spec.nct_id
        if not r.dataset:
            patch["dataset"] = cohort.dataset
        return r.model_copy(update=patch) if patch else r
    if not hasattr(r, "point_estimate"):  # already the contract-shaped object
        return r
    measure = _MEASURE.get(r.effect_measure or "", EffectMeasure.OR)
    est = r.point_estimate if r.point_estimate is not None else float("nan")
    return TTEResult(
        nct_id=spec.nct_id, dataset=cohort.dataset,
        method=(r.adjustment or adjustment), measure=measure, estimate=est,
        ci_low=r.ci_low, ci_high=r.ci_high, n_treated=r.n_treated, n_control=r.n_control,
        extra={
            "p_value": r.p_value, "abs_risk_diff": r.abs_risk_diff, "nnt": r.nnt,
            "e_value_point": r.e_value_point, "e_value_ci": r.e_value_ci,
            "ok": r.ok, "error": r.error, "test": r.test,
        },
    )


def make_engine_provider(
    covariates: "list[FeatureSpec] | None" = None,
    *,
    adjustment: str = "iptw",
    outcome_kind: str = "binary",
) -> Callable:
    """Return a Pipeline `tte` provider that runs the #10 engine.

    `covariates` are the FeatureSpecs adjusted for (the confounders); they are
    materialized into the analysis frame and passed to the estimator.
    """
    covariates = covariates or []

    def provider(events, cohort, spec) -> TTEResult:
        from tteEngine.analysis import (add_treatment_indicator, outcome_column,
                                        run_tte, select_measurable_outcome)
        from tteEngine.cohort import build_analysis_frame

        treated = next((a.name for a in cohort.arms if not a.is_control), None)
        if treated is None or not spec.outcomes:
            return TTEResult(nct_id=spec.nct_id, dataset=cohort.dataset, method=adjustment,
                             measure=EffectMeasure.RR, estimate=float("nan"),
                             extra={"error": "no treated arm or no outcome"})
        frame = build_analysis_frame(events, cohort, spec, covariates=covariates)
        frame = add_treatment_indicator(frame, group_col="group", treated_value=treated)
        # #146: emulate the MEASURABLE outcome (prefer binary mortality), not blindly
        # spec.outcomes[0] (often a non-mortality ctgov endpoint -> KeyError/empty drop).
        outcome = select_measurable_outcome(spec, frame.columns)
        if outcome is None:
            return TTEResult(nct_id=spec.nct_id, dataset=cohort.dataset, method=adjustment,
                             measure=EffectMeasure.RR, estimate=float("nan"),
                             extra={"error": "no measurable outcome in dataset"})
        outcome_col = outcome_column(outcome.name)
        frame[outcome_col] = frame[outcome_col].astype(int)
        r = run_tte(frame, outcome_col=outcome_col, covariates=[c.name for c in covariates],
                    adjustment=adjustment, outcome_kind=outcome_kind)
        return _to_contract(r, spec, cohort, adjustment)

    return provider
