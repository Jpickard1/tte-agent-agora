"""Emulated-vs-observed comparison (#11, probe / lane:analysis).

Parse a ClinicalTrials.gov study's *posted results* into the trial's reported
treatment-vs-control contrast, and judge whether the emulated estimate (a
contracts.TTEResult from #10) agrees in direction — the heart of "emulate the
trial in context". Returns the canonical contracts.ComparisonResult.

Self-contained PORT of trialsim app/trialsim/compare.py (per jpic: no cross-repo
imports). Reads the RAW ctgov resultsSection.outcomeMeasures (#1 reader output);
token-based arm matching (no trialsim.concepts dependency).
"""
from __future__ import annotations

import math
import re

from ..contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult
from ..ctgov.reader import nct_id_of, reported_outcome_measures

_CONTROL_RE = re.compile(
    r"\b(placebo|control|usual|standard|conventional|comparator|saline|"
    r"sham|no treatment|routine|best supportive)\b",
    re.I,
)


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _measurements(om: dict) -> dict[str, float]:
    classes = om.get("classes") or []
    if not classes:
        return {}
    cats = classes[0].get("categories") or []
    if not cats:
        return {}
    return {m["groupId"]: _num(m.get("value"))
            for m in (cats[0].get("measurements") or [])
            if m.get("groupId") and _num(m.get("value")) is not None}


def _denoms(om: dict) -> dict[str, float]:
    for d in om.get("denoms", []) or []:
        out = {c["groupId"]: _num(c.get("value"))
               for c in (d.get("counts") or [])
               if c.get("groupId") and _num(c.get("value")) is not None}
        if out:
            return out
    return {}


def _is_count(om: dict) -> bool:
    return (om.get("paramType") or "").upper().startswith("COUNT") or bool(om.get("denoms"))


def _title_match_score(title: str, target: str) -> tuple:
    """How well a reported outcome title matches the EMULATED outcome name: both
    being mortality endpoints ranks highest, then shared significant-word overlap."""
    from .outcomes import is_mortality_outcome
    tw = {w for w in re.findall(r"[a-z]+", (title or "").lower()) if len(w) > 3}
    gw = {w for w in re.findall(r"[a-z]+", (target or "").lower()) if len(w) > 3}
    return (is_mortality_outcome(title) and is_mortality_outcome(target), len(tw & gw))


def _pick_count_for(outcome_measures, outcome_name: str | None = None) -> dict | None:
    """Pick the reported count outcome to compare against. When `outcome_name` (the
    EMULATED outcome) is given, prefer the reported outcome whose title matches it
    (like-for-like comparison); otherwise fall back to the PRIMARY count outcome."""
    cands = [om for om in (outcome_measures or [])
             if _is_count(om) and len(_measurements(om)) >= 2]
    if not cands:
        return None
    if outcome_name:
        best = max(cands, key=lambda om: _title_match_score(om.get("title", ""), outcome_name))
        if _title_match_score(best.get("title", ""), outcome_name) != (False, 0):
            return best   # a real title match -> align to it
    prim = [o for o in cands if (o.get("type") or "").upper() == "PRIMARY"]
    return (prim or cands)[0]


def _pick_primary_count(outcome_measures) -> dict | None:
    return _pick_count_for(outcome_measures, None)


def parse_reported_effect(study: dict, treatment_hint: str = "",
                          outcome_name: str | None = None) -> dict | None:
    """Extract a count outcome's treatment-vs-control contrast from the study's
    posted results. `outcome_name` (the emulated outcome) aligns the reported effect
    to the SAME endpoint (like-for-like); else the primary count outcome. Returns a
    dict (always with per-arm ``rates``; ``effect`` only when both arms found) or None."""
    om = _pick_count_for(reported_outcome_measures(study), outcome_name)
    if om is None:
        return None
    titles = {g.get("id"): g.get("title", "") for g in om.get("groups", []) or []}
    vals, denoms = _measurements(om), _denoms(om)
    rates = [{"group": t, "groupId": gid, "value": vals.get(gid), "denom": denoms.get(gid),
              "pct": (100.0 * vals[gid] / denoms[gid]) if (gid in vals and denoms.get(gid)) else None}
             for gid, t in titles.items()]
    base = {"title": om.get("title", ""), "time_frame": om.get("timeFrame", ""),
            "rates": rates, "effect": None}

    arms = [r for r in rates if r["pct"] is not None]
    if len(arms) < 2:
        return base
    hint = {w for w in re.findall(r"[a-z]+", (treatment_hint or "").lower()) if len(w) > 3}
    control = next((a for a in arms if _CONTROL_RE.search(a["group"] or "")), None)
    treated = next((a for a in arms if any(w in (a["group"] or "").lower() for w in hint)), None)
    if control is not None and treated is None and len(arms) == 2:
        treated = next(a for a in arms if a is not control)
    if treated is not None and control is None and len(arms) == 2:
        control = next(a for a in arms if a is not treated)
    if treated is None or control is None or treated is control:
        return base

    p_t, p_c = treated["pct"] / 100.0, control["pct"] / 100.0
    rd = p_t - p_c
    base["effect"] = {
        "treated_arm": treated["group"], "control_arm": control["group"],
        "p_treated": treated["pct"], "p_control": control["pct"],
        "risk_diff": rd, "risk_ratio": (p_t / p_c) if p_c > 0 else None,
        "direction": "higher" if rd > 0 else ("lower" if rd < 0 else "equal"),
    }
    return base


def _null_value(measure: EffectMeasure) -> float:
    return 0.0 if measure == EffectMeasure.RD else 1.0


def _harm(estimate: float, measure: EffectMeasure) -> bool:
    """Treatment increases the (harmful) outcome: ratio>1, or RD>0."""
    return estimate > _null_value(measure)


def _spans_null(emulated: TTEResult) -> bool:
    null = _null_value(emulated.measure)
    return (emulated.ci_low is not None and emulated.ci_high is not None
            and emulated.ci_low <= null <= emulated.ci_high)


def _observed(effect: dict) -> tuple[float | None, EffectMeasure | None]:
    """Reduce the reported contrast to (estimate, measure): prefer RR, else RD."""
    if not effect:
        return None, None
    if effect.get("risk_ratio") is not None:
        return float(effect["risk_ratio"]), EffectMeasure.RR
    if effect.get("risk_diff") is not None:
        return float(effect["risk_diff"]), EffectMeasure.RD
    return None, None


def _judge(effect: dict, emulated: TTEResult) -> tuple[Agreement, str]:
    obs_est, obs_meas = _observed(effect)
    est = emulated.estimate
    if obs_est is None or est is None or not math.isfinite(est):
        return Agreement.INCONCLUSIVE, "Need a reported contrast and a finite emulated estimate."
    rep_harm = _harm(obs_est, obs_meas)
    rep_null = abs(obs_est - _null_value(obs_meas)) < (0.05 if obs_meas != EffectMeasure.RD else 1e-9)
    rep_str = f"{obs_meas.value} {obs_est:.2f}"
    if rep_null or _spans_null(emulated):
        return Agreement.INCONCLUSIVE, (
            "The emulated CI spans the null, " if _spans_null(emulated)
            else "The reported effect is near null, ") + "so direction can't be firmly compared."
    if rep_harm == _harm(est, emulated.measure):
        d = "increased" if _harm(est, emulated.measure) else "reduced"
        return Agreement.CONCORDANT, (
            f"Both trial and emulation show treatment {d} the outcome "
            f"(reported {rep_str}, emulated {emulated.measure.value} {est:.2f}).")
    return Agreement.DISCORDANT, (
        f"Directions disagree: trial reported {rep_str} but emulation estimated {est:.2f}. "
        f"Expected with residual confounding, population differences, or outcome-proxy mismatch.")


def compare_trial(study: dict, emulated: TTEResult, *, treatment_hint: str = "",
                  dataset: str | None = None) -> ComparisonResult:
    """One emulated-vs-observed row (contracts.ComparisonResult): parse the trial's
    reported effect for the SAME outcome the engine emulated (like-for-like) and judge
    directional agreement with the emulated TTEResult."""
    emulated_outcome = (emulated.extra or {}).get("outcome")
    effect = (parse_reported_effect(study, treatment_hint,
                                    outcome_name=emulated_outcome) or {}).get("effect") or {}
    obs_est, obs_meas = _observed(effect)
    agreement, note = _judge(effect, emulated)
    return ComparisonResult(
        nct_id=emulated.nct_id or (nct_id_of(study) or ""),
        dataset=dataset or emulated.dataset or "",
        emulated=emulated,
        observed_estimate=obs_est,
        observed_measure=obs_meas,
        agreement=agreement,
        notes=note,
    )
