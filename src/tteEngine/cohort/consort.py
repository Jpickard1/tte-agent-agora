"""CONSORT-style attrition + cohort diagnostics report (#29).

Renders the #30 CohortDiagnostics as the target-trial-emulation attrition flow

    screened -> (excluded: ineligible) -> eligible
             -> (excluded: immortal-time) -> enrolled -> per-arm

per trial x dataset, and aggregates it across the corpus for the gallery. Pure
(no pandas) — operates on the typed diagnostics, so it's import-light and feeds
both the single-trial report and #36's batch summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from tteEngine.contracts.cohort import CohortDiagnostics


def consort_flow(diag: "CohortDiagnostics") -> dict:
    """Structured attrition flow with explicit exclusion counts at each step.
    `excluded_ineligible` + `excluded_immortal_time` + `enrolled` reconcile to
    `screened`, so nothing is unaccounted for."""
    ineligible = max(diag.n_screened - diag.n_eligible, 0)
    return {
        "screened": diag.n_screened,
        "excluded_ineligible": ineligible,
        "eligible": diag.n_eligible,
        "excluded_immortal_time": diag.n_excluded_immortal,
        "enrolled": diag.n_enrolled,
        "arms": dict(diag.arm_sizes),
        "anchor": diag.anchor,
        "landmark_hours": diag.landmark_hours,
        "leakage_warnings": list(diag.leakage_warnings),
    }


def format_consort(diag: "CohortDiagnostics", *, title: str = "Cohort attrition") -> str:
    """Human-readable CONSORT-style diagram for one cohort."""
    f = consort_flow(diag)
    lines = [
        f"{title}  (t0 anchor={f['anchor']}, landmark={f['landmark_hours']}h)",
        f"  screened                       {f['screened']}",
        f"    -- excluded (ineligible)      -{f['excluded_ineligible']}",
        f"  eligible                       {f['eligible']}",
        f"    -- excluded (immortal-time)   -{f['excluded_immortal_time']}",
        f"  enrolled                       {f['enrolled']}",
    ]
    for arm, n in f["arms"].items():
        lines.append(f"      |- {arm}: {n}")
    for w in f["leakage_warnings"]:
        lines.append(f"  [!] {w}")
    return "\n".join(lines)


def aggregate_diagnostics(diags: Iterable["CohortDiagnostics"]) -> dict:
    """Corpus-level attrition: sum the flow across many cohorts (for the #36
    gallery). Tracks how many cohorts carried an eligibility-leakage warning so
    data-quality issues surface at scale rather than per-trial only."""
    agg = {
        "n_cohorts": 0, "screened": 0, "eligible": 0,
        "excluded_immortal": 0, "enrolled": 0, "n_cohorts_with_leakage": 0,
    }
    for d in diags:
        agg["n_cohorts"] += 1
        agg["screened"] += d.n_screened
        agg["eligible"] += d.n_eligible
        agg["excluded_immortal"] += d.n_excluded_immortal
        agg["enrolled"] += d.n_enrolled
        if d.leakage_warnings:
            agg["n_cohorts_with_leakage"] += 1
    return agg
