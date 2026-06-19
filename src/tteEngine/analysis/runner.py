"""Typed TTE engine entrypoint (#10, probe / lane:analysis).

Runs the ported causal engine (engine.py) over an analysis-ready cohort frame
(the #9 seam from tteEngine.cohort.build_analysis_frame) and returns a typed
``TTEResult`` — the #10 -> #11 seam the emulated-vs-observed benchmark consumes.

The heavy estimators live in engine.py (ported, self-contained). This module is
the thin, typed boundary: build the engine's OutcomeSpec/AnalysisConfig from
simple arguments, call run_analysis, and map the result dict to a stable schema.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field


class BalanceRow(BaseModel):
    """Standardized mean difference for one covariate, before/after adjustment."""

    variable: str
    smd_before: float
    smd_after: float


class TTEResult(BaseModel):
    """Typed emulated effect estimate + diagnostics (the #10 -> #11 seam)."""

    ok: bool
    outcome: str
    effect_measure: str | None = None  # 'Hazard Ratio' | 'Odds Ratio' | 'Risk Difference'
    point_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    n_analyzed: int = 0
    n_treated: int = 0
    n_control: int = 0
    abs_risk_diff: float | None = None
    nnt: float | None = None
    nnt_kind: str | None = None
    e_value_point: float | None = None
    e_value_ci: float | None = None
    adjustment: str | None = None
    n_unbalanced_before: int | None = None
    n_unbalanced_after: int | None = None
    balance: list[BalanceRow] = Field(default_factory=list)
    test: str | None = None
    error: str | None = None


def _f(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _balance_rows(b: dict | None) -> list[BalanceRow]:
    b = b or {}
    before, after = b.get("before"), b.get("after")
    if before is None:
        return []
    covars = b.get("covars") or list(before.index)
    rows: list[BalanceRow] = []
    for cov in covars:
        rows.append(
            BalanceRow(
                variable=str(cov),
                smd_before=_f(before.get(cov)) if _f(before.get(cov)) is not None else float("nan"),
                smd_after=(_f(after.get(cov)) if after is not None and _f(after.get(cov)) is not None
                           else float("nan")),
            )
        )
    return rows


def add_treatment_indicator(frame, *, group_col: str = "group", out_col: str = "T",
                            treated_value=None, control_value=None):
    """Add a 0/1 treatment column from an arm/group label column. Specify the
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
) -> TTEResult:
    """Estimate the emulated treatment effect on an analysis-ready cohort frame.

    `frame` must have a 0/1 `treatment_col`, the `covariates`, and the outcome
    column(s): `outcome_col` (binary) or (`outcome_col` event + `time_col`) for
    survival. Adjustment: 'iptw' | 'psm' | 'covariate' | 'unadjusted'.
    """
    # lazy: keep `import tteEngine.analysis` light — the heavy estimators (and the
    # lifelines/statsmodels `analysis` extra) only load when an analysis is run.
    from .engine import AnalysisConfig, OutcomeSpec, run_analysis

    spec = OutcomeSpec(
        key=outcome_col, label=label or outcome_col, kind=outcome_kind,
        event_col=outcome_col, time_col=time_col, horizon_days=horizon_days,
        reverse=reverse,
    )
    cfg = AnalysisConfig(
        treatment_col=treatment_col, covariates=list(covariates), adjustment=adjustment,
        survival_test=survival_test, binary_test=binary_test, id_col=id_col,
    )
    res = run_analysis(frame, cfg, spec)
    if not res.get("ok"):
        return TTEResult(ok=False, outcome=spec.label, error=res.get("error"),
                         n_analyzed=int(res.get("n_analyzed", 0)), adjustment=adjustment)
    ev = res.get("e_value") or {}
    return TTEResult(
        ok=True,
        outcome=res.get("outcome", spec.label),
        effect_measure=res.get("estimate_name"),
        point_estimate=_f(res.get("estimate")),
        ci_low=_f(res.get("ci_low")),
        ci_high=_f(res.get("ci_high")),
        p_value=_f(res.get("p_value")),
        n_analyzed=int(res.get("n_analyzed", 0)),
        n_treated=int(res.get("n_treated", 0)),
        n_control=int(res.get("n_control", 0)),
        abs_risk_diff=_f(res.get("abs_risk_diff")),
        nnt=_f(res.get("nnt")),
        nnt_kind=res.get("nnt_kind"),
        e_value_point=_f(ev.get("point")),
        e_value_ci=_f(ev.get("ci")),
        adjustment=res.get("adjustment", adjustment),
        n_unbalanced_before=res.get("n_unbalanced_before"),
        n_unbalanced_after=res.get("n_unbalanced_after"),
        balance=_balance_rows(res.get("balance")),
        test=res.get("test"),
    )
