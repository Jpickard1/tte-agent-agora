"""Emulated-vs-observed comparison (#11, probe / lane:analysis).

Parse a ClinicalTrials.gov study's *posted results* into the trial's reported
treatment-vs-control contrast, and judge whether the emulated estimate (a #10
TTEResult) agrees in direction — the heart of "emulate the trial in context".

Self-contained PORT of trialsim app/trialsim/compare.py (per jpic: no cross-repo
imports). Differences: (1) reads the RAW ctgov resultsSection.outcomeMeasures
(what the #1 reader returns) directly, instead of a pre-parsed format; (2) the
drug-name arm matcher is token-based (no trialsim.concepts dependency).
"""
from __future__ import annotations

import re

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
    """{groupId: value} from the first class/category's measurements."""
    classes = om.get("classes") or []
    if not classes:
        return {}
    cats = classes[0].get("categories") or []
    if not cats:
        return {}
    out: dict[str, float] = {}
    for m in cats[0].get("measurements", []) or []:
        v = _num(m.get("value"))
        if v is not None and m.get("groupId"):
            out[m["groupId"]] = v
    return out


def _denoms(om: dict) -> dict[str, float]:
    """{groupId: denominator} from the first non-empty denoms block."""
    for d in om.get("denoms", []) or []:
        out = {c["groupId"]: _num(c.get("value"))
               for c in (d.get("counts") or []) if c.get("groupId") and _num(c.get("value")) is not None}
        if out:
            return out
    return {}


def _is_count(om: dict) -> bool:
    return (om.get("paramType") or "").upper().startswith("COUNT") or bool(om.get("denoms"))


def _pick_primary_count(outcome_measures) -> dict | None:
    """First PRIMARY participant-count outcome with >=2 arms carrying a value;
    fall back to any count outcome, else None."""
    cands = [om for om in (outcome_measures or [])
             if _is_count(om) and len(_measurements(om)) >= 2]
    if not cands:
        return None
    prim = [o for o in cands if (o.get("type") or "").upper() == "PRIMARY"]
    return (prim or cands)[0]


def parse_reported_effect(study: dict, treatment_hint: str = "") -> dict | None:
    """Extract the primary count outcome's treatment-vs-control contrast from the
    study's posted results. Returns a dict (always with per-arm ``rates``;
    ``effect`` populated only when both arms are identified) or None."""
    om = _pick_primary_count(reported_outcome_measures(study))
    if om is None:
        return None
    titles = {g.get("id"): g.get("title", "") for g in om.get("groups", []) or []}
    vals, denoms = _measurements(om), _denoms(om)
    rates = []
    for gid, title in titles.items():
        v, d = vals.get(gid), denoms.get(gid)
        rates.append({"group": title, "groupId": gid, "value": v, "denom": d,
                      "pct": (100.0 * v / d) if (v is not None and d) else None})
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
        "risk_diff_pp": 100.0 * rd, "risk_ratio": (p_t / p_c) if p_c > 0 else None,
        "direction": "higher" if rd > 0 else ("lower" if rd < 0 else "equal"),
    }
    return base


def _as_emulated(emulated) -> dict:
    """Accept a #10 TTEResult or a plain dict."""
    if hasattr(emulated, "point_estimate"):
        return {"estimate": emulated.point_estimate, "ci_low": emulated.ci_low,
                "ci_high": emulated.ci_high, "estimate_name": emulated.effect_measure}
    return emulated or {}


def concordance(reported_effect: dict, emulated, benefit_lower: bool = True) -> dict:
    """Judge directional agreement between the trial's reported contrast and the
    emulated estimate (HR/OR vs the no-effect value 1). Returns {badge, verdict,
    note}. Accepts a TTEResult or a dict for `emulated`."""
    eff = reported_effect or {}
    emu = _as_emulated(emulated)
    rr, rd_pp, est = eff.get("risk_ratio"), eff.get("risk_diff_pp"), emu.get("estimate")
    if (rr is None and rd_pp is None) or est is None:
        return {"badge": "—", "verdict": "not comparable",
                "note": "Need both a reported contrast and an emulated estimate."}
    if rr is not None:
        rep_harm, rep_null, rep_str = rr > 1.0, abs(rr - 1.0) < 0.05, f"RR {rr:.2f}"
    else:
        rep_harm, rep_null, rep_str = rd_pp > 0, abs(rd_pp) < 1e-9, f"RD {rd_pp:+.1f} pp"
    emu_harm = est > 1.0
    ci_lo, ci_hi = emu.get("ci_low"), emu.get("ci_high")
    emu_null = ci_lo is not None and ci_hi is not None and ci_lo <= 1.0 <= ci_hi
    if rep_null or emu_null:
        return {"badge": "🟰", "verdict": "inconclusive",
                "note": ("The emulated CI spans 1 (no significant effect), " if emu_null
                         else "The trial's reported effect is near null, ")
                + "so direction can't be firmly compared."}
    if rep_harm == emu_harm:
        d = "increased" if emu_harm else "reduced"
        return {"badge": "✅", "verdict": "concordant",
                "note": f"Both trial and emulation show treatment {d} the outcome "
                        f"(reported {rep_str}, emulated {emu.get('estimate_name', 'est')} {est:.2f})."}
    return {"badge": "⚠️", "verdict": "discordant",
            "note": f"Directions disagree: trial reported {rep_str} but emulation estimated "
                    f"{est:.2f}. Expected with residual confounding, population differences, "
                    f"or an outcome-proxy mismatch."}


def compare_trial(study: dict, tte_result, *, treatment_hint: str = "", dataset: str | None = None) -> dict:
    """One emulated-vs-observed row: parse the trial's reported effect, judge
    concordance with the emulated TTEResult."""
    rep = parse_reported_effect(study, treatment_hint)
    emu = _as_emulated(tte_result)
    verdict = concordance((rep or {}).get("effect") or {}, tte_result)
    return {
        "nct_id": nct_id_of(study),
        "dataset": dataset,
        "reported_effect": (rep or {}).get("effect"),
        "emulated": emu,
        "verdict": verdict["verdict"],
        "badge": verdict["badge"],
        "note": verdict["note"],
    }
