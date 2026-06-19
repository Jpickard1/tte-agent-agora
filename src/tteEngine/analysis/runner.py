"""Typed TTE engine entrypoint (#10, probe / lane:analysis).

Runs the ported causal engine (engine.py) over an analysis-ready cohort frame
(the #9 seam from tteEngine.cohort.build_analysis_frame) and returns the single
canonical seam type, ``tteEngine.contracts.results.TTEResult`` — what the spine
(#9/#12/#13) and the benchmark (#11) consume. Rich method-specific diagnostics
(p-value, balance, E-values, NNT) ride in ``.extra``.

Import-safe: the seam TYPE lives in import-light contracts/; the heavy estimators
(lifelines/statsmodels, the `analysis` extra) load lazily only when run_tte is
actually called — so the orchestration can degrade gracefully without the extra.
"""
from __future__ import annotations

import math

from ..contracts.results import EffectMeasure, TTEResult

_MEASURE = {
    "Hazard Ratio": EffectMeasure.HR,
    "Odds Ratio": EffectMeasure.OR,
    "Risk Difference": EffectMeasure.RD,
    "Risk Ratio": EffectMeasure.RR,
}


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _balance_rows(b: dict | None) -> list[dict]:
    b = b or {}
    before, after = b.get("before"), b.get("after")
    if before is None:
        return []
    covars = b.get("covars") or list(before.index)
    return [
        {"variable": str(cov), "smd_before": _f(before.get(cov)),
         "smd_after": _f(after.get(cov)) if after is not None else None}
        for cov in covars
    ]


def add_treatment_indicator(frame, *, group_col: str = "group", out_col: str = "T",
                            treated_value=None, control_value=None):
    """Add a 0/1 treatment column from an arm/group label column. Specify
    treated_value (preferred) or control_value; with neither, the second of two
    sorted group values is treated as the treatment arm."""
    f = frame.copy()
    if treated_value is not None:
        f[out_col] = (f[group_col] == treated_value).astype(int)
    elif control_value is not None:
        f[out_col] = (f[group_col] != control_value).astype(int)
    else:
        vals = sorted(v for v in f[group_col].dropna().unique())
        if len(vals) != 2:
            raise ValueError("specify treated_value/control_value for non-binary group column")
        f[out_col] = (f[group_col] == vals[-1]).astype(int)
    return f


def run_tte(
    frame,
    *,
    outcome_col: str,
    covariates,
    treatment_col: str = "T",
    outcome_kind: str = "binary",
    time_col: str | None = None,
    adjustment: str = "iptw",
    survival_test: str = "cox",
    binary_test: str = "logistic",
    label: str | None = None,
    horizon_days: float = 28.0,
    id_col: str = "TRAJECTORY_ID",
    reverse: bool = False,
    restrict_col: str | None = None,
    nct_id: str = "",
    dataset: str = "",
) -> TTEResult:
    """Estimate the emulated treatment effect on an analysis-ready cohort frame
    and return a contracts.TTEResult. `frame` needs a 0/1 `treatment_col`, the
    `covariates`, and the outcome column(s). Adjustment: iptw|psm|covariate|unadjusted.
    `restrict_col`: analyse only rows where it == 1 (e.g. an adherence flag for a
    per-protocol estimand)."""
    # lazy: importing tteEngine.analysis stays light; lifelines/statsmodels load
    # only here, when an analysis is actually run.
    from .engine import AnalysisConfig, OutcomeSpec, run_analysis

    spec = OutcomeSpec(
        key=outcome_col, label=label or outcome_col, kind=outcome_kind,
        event_col=outcome_col, time_col=time_col, horizon_days=horizon_days,
        reverse=reverse, restrict_col=restrict_col,
    )
    cfg = AnalysisConfig(
        treatment_col=treatment_col, covariates=list(covariates), adjustment=adjustment,
        survival_test=survival_test, binary_test=binary_test, id_col=id_col,
    )
    res = run_analysis(frame, cfg, spec)
    measure = _MEASURE.get(res.get("estimate_name"), EffectMeasure.OR)
    method = res.get("adjustment") or adjustment

    if not res.get("ok"):
        return TTEResult(
            nct_id=nct_id, dataset=dataset, method=method, measure=measure,
            estimate=float("nan"),
            extra={"ok": False, "error": res.get("error"), "outcome": spec.label,
                   "n_analyzed": int(res.get("n_analyzed", 0))},
        )

    est = _f(res.get("estimate"))
    ev = res.get("e_value") or {}
    bal = res.get("balance") or {}
    return TTEResult(
        nct_id=nct_id,
        dataset=dataset,
        method=method,
        measure=measure,
        estimate=est if est is not None else float("nan"),
        ci_low=_f(res.get("ci_low")),
        ci_high=_f(res.get("ci_high")),
        n_treated=int(res.get("n_treated", 0)),
        n_control=int(res.get("n_control", 0)),
        extra={
            "ok": True,
            "outcome": res.get("outcome", spec.label),
            "effect_measure": res.get("estimate_name"),
            "p_value": _f(res.get("p_value")),
            "abs_risk_diff": _f(res.get("abs_risk_diff")),
            "nnt": _f(res.get("nnt")),
            "nnt_kind": res.get("nnt_kind"),
            "e_value_point": _f(ev.get("point")),
            "e_value_ci": _f(ev.get("ci")),
            "n_analyzed": int(res.get("n_analyzed", 0)),
            "n_unbalanced_before": res.get("n_unbalanced_before"),
            "n_unbalanced_after": res.get("n_unbalanced_after"),
            "balance": _balance_rows(res.get("balance")),
            # #105: surface the covariates actually in the model + the PS-overlap /
            # common-support diagnostic (computed by the engine but previously dropped),
            # so the confounder-adjustability ledger + UI can read them from the corpus.
            "covariates_used": list(bal.get("covars") or []),
            "ps_overlap": bal.get("overlap"),
            "test": res.get("test"),
        },
    )
