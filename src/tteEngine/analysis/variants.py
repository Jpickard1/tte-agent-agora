"""Estimand variants + prespecified subgroups (#38, probe / lane:analysis).

- Estimand variants: ITT (analyse by assigned arm) vs per-protocol (restrict to
  adherent rows via a flag column). Selectable per trial.
- Prespecified subgroups / effect modification: re-run the emulation within each
  level of a subgroup column + report an effect-modification (heterogeneity) flag.

Both build on run_tte (the engine handles restriction + subgroup masking); results
are reported as typed objects carrying contracts.TTEResult.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..contracts.results import TTEResult
from .runner import run_tte


def run_estimand_variants(frame, *, outcome_col, covariates, adherence_col: str | None = None,
                          **run_tte_kwargs) -> dict[str, TTEResult]:
    """Return {estimand: TTEResult}: 'itt' always; 'per_protocol' when an
    adherence flag column is given (restricts to rows where it == 1)."""
    out = {"itt": run_tte(frame, outcome_col=outcome_col, covariates=covariates, **run_tte_kwargs)}
    if adherence_col is not None:
        out["per_protocol"] = run_tte(frame, outcome_col=outcome_col, covariates=covariates,
                                      restrict_col=adherence_col, **run_tte_kwargs)
    return out


class SubgroupEstimate(BaseModel):
    subgroup: str
    n_analyzed: int
    result: TTEResult


class SubgroupReport(BaseModel):
    subgroup_col: str
    overall: TTEResult
    subgroups: list[SubgroupEstimate] = Field(default_factory=list)
    heterogeneity: bool = False
    note: str = ""


def _ci_overlap(a: TTEResult, b: TTEResult) -> bool:
    if None in (a.ci_low, a.ci_high, b.ci_low, b.ci_high):
        return True  # incomparable -> don't claim modification
    return not (a.ci_high < b.ci_low or b.ci_high < a.ci_low)


def run_subgroups(frame, *, subgroup_col, outcome_col, covariates, **run_tte_kwargs) -> SubgroupReport:
    """Prespecified subgroup analysis: run_tte within each level of subgroup_col +
    overall. heterogeneity = any pair of subgroup CIs are disjoint (a simple,
    deterministic effect-modification signal; a formal interaction test can layer on)."""
    overall = run_tte(frame, outcome_col=outcome_col, covariates=covariates, **run_tte_kwargs)
    subs: list[SubgroupEstimate] = []
    for val in sorted(frame[subgroup_col].dropna().unique()):
        sub = frame[frame[subgroup_col] == val]
        r = run_tte(sub, outcome_col=outcome_col, covariates=covariates, **run_tte_kwargs)
        subs.append(SubgroupEstimate(subgroup=f"{subgroup_col}={val}",
                                     n_analyzed=r.n_treated + r.n_control, result=r))
    ok = [s.result for s in subs if s.result.extra.get("ok", True)]
    het = any(not _ci_overlap(ok[i], ok[j]) for i in range(len(ok)) for j in range(i + 1, len(ok)))
    return SubgroupReport(
        subgroup_col=subgroup_col, overall=overall, subgroups=subs, heterogeneity=het,
        note=(f"{len(subs)} subgroups; effect modification "
              f"{'DETECTED (non-overlapping CIs)' if het else 'not detected'}."),
    )
