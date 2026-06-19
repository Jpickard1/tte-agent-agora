"""Missingness + proxy-substitution report (#34, worker1).

Three questions per (trial, dataset), building on #33's measurability:

  1. PROXY LIST   — which protocol elements were satisfied by a surrogate rather
     than the exact protocol variable (= #33's PROXY-classified elements).
  2. MISSINGNESS  — for the variables we DO extract, how much is actually present
     in the built cohort (per-feature missing fraction over the analysis frame).
  3. SENSITIVITY  — does the estimate move (or the conclusion flip) when a
     proxy-substituted adjustment variable is dropped? i.e. how much do we lean
     on the proxy.

(1) is pure (spec-only). (2)/(3) need a built cohort/frame (pandas) and are
guarded so the module imports in CI's [dev] env. Together they answer #33's sibling
question: not just *can* we measure it, but *how much is really there* and *how
much does the answer depend on the proxies*.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts.trial_spec import TargetTrialSpec
from .measurability import PROXY, measurability_report

if TYPE_CHECKING:
    import pandas as pd

_NON_FEATURE_COLS = ("TRAJECTORY_ID", "group", "time_zero")


def proxy_substitution_list(spec: TargetTrialSpec, dataset: str) -> list[dict]:
    """The proxy variables used for a trial in a dataset (#33's PROXY elements):
    each protocol element matched by a surrogate / pending adapter wiring."""
    rep = measurability_report(spec, dataset)
    return [{"kind": e.kind, "concept": e.concept, "event_type": e.event_type, "reason": e.reason}
            for e in rep.elements if e.status == PROXY]


def missingness_summary(frame: "pd.DataFrame", columns: list[str] | None = None) -> dict:
    """Per-feature missingness over an analysis WIDE frame (build_analysis_frame
    output). Returns counts + fraction per column, the overall mean, and the worst
    column — the data-availability view behind the measurability classification."""
    cols = columns if columns is not None else [c for c in frame.columns if c not in _NON_FEATURE_COLS]
    n = len(frame)
    per = {}
    for c in cols:
        n_missing = int(frame[c].isna().sum())
        per[c] = {"n": n, "n_missing": n_missing,
                  "missing_fraction": round(n_missing / n, 4) if n else 0.0}
    worst = max(per.items(), key=lambda kv: kv[1]["missing_fraction"])[0] if per else None
    overall = round(sum(v["missing_fraction"] for v in per.values()) / len(per), 4) if per else 0.0
    return {"n_rows": n, "n_features": len(per), "columns": per,
            "mean_missing_fraction": overall, "worst_feature": worst}


def proxy_sensitivity(estimate_full: float, estimate_reduced: float, *,
                      null: float = 1.0) -> dict:
    """How much the estimate moves when a proxy-substituted adjustment variable is
    dropped (full = adjusted incl. the proxy; reduced = without it). Flags whether
    the qualitative conclusion (which side of the null) FLIPS — i.e. whether the
    finding leans on the proxy."""
    def side(e: float) -> str:
        return "benefit" if e < null else ("harm" if e > null else "null")

    abs_delta = abs(estimate_reduced - estimate_full)
    flips = side(estimate_full) != side(estimate_reduced)
    return {
        "estimate_full": estimate_full,
        "estimate_reduced": estimate_reduced,
        "abs_delta": round(abs_delta, 4),
        "rel_delta": round(abs_delta / abs(estimate_full), 4) if estimate_full else None,
        "conclusion_full": side(estimate_full),
        "conclusion_reduced": side(estimate_reduced),
        "robust_to_proxy": not flips,
    }


def missingness_and_proxy_report(
    spec: TargetTrialSpec, dataset: str, *,
    frame: "pd.DataFrame | None" = None, feature_columns: list[str] | None = None,
    sensitivity: dict | list[dict] | None = None,
) -> dict:
    """The combined #34 report: proxy list (always) + missingness (if a built
    analysis `frame` is given) + proxy sensitivity (if precomputed via
    `proxy_sensitivity`). Acceptance: missingness summary + proxy list +
    sensitivity-to-proxies, per trial-per-DB."""
    proxies = proxy_substitution_list(spec, dataset)
    out: dict = {"nct_id": spec.nct_id, "dataset": dataset,
                 "proxy_list": proxies, "n_proxies": len(proxies)}
    if frame is not None:
        out["missingness"] = missingness_summary(frame, feature_columns)
    if sensitivity is not None:
        out["proxy_sensitivity"] = sensitivity
    return out


__all__ = [
    "proxy_substitution_list", "missingness_summary", "proxy_sensitivity",
    "missingness_and_proxy_report",
]
