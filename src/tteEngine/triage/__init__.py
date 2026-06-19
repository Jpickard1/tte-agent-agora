"""Emulability triage (#35): score which ctgov trials are emulable in each ICU
dataset (MIMIC-IV / eICU-CRD) and produce a ranked, sortable catalog.

A trial is *emulable* in a dataset when its eligibility, exposure (arm
interventions) and outcome can be reconstructed from that dataset's events. This
module scores each TargetTrialSpec (#2's output) per dataset from the spec's
structure + the vocab layer (#5):

  - eligibility: criteria whose EVENT_TYPE is captured by ICU EHR (diagn/lab/
    medic/measu/proce/demog) and whose concept resolves;
  - exposure:    at least one treatment arm with an identifiable intervention;
  - outcome:     a within-horizon, EHR-measurable endpoint (mortality is the
    canonical one; very long horizons / non-EHR endpoints score down).

Directive (jpic): MAXIMIZE count — never silently cap. EVERY trial gets a row with
its score + reasons (low-emulability trials are flagged, NOT dropped), so nothing
is hidden. Sepsis trials are prioritized (flagged + sorted first).

v1 heuristics are deliberately simple + documented; refine against the real
MIMIC/eICU data dictionaries later. Pure (no pandas) — operates on specs.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from tteEngine import vocab
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import TargetTrialSpec

#: event domains ICU EHR (MIMIC-IV / eICU-CRD) actually captures.
EHR_EVENT_TYPES: set[EventType] = {
    EventType.DIAGNOSIS, EventType.LAB, EventType.MEDICATION,
    EventType.MEASUREMENT, EventType.PROCEDURE, EventType.DEMOGRAPHIC,
    EventType.OUTCOME, EventType.LOCATION,
}

#: outcomes reliably measurable in ICU data, by concept keyword.
MEASURABLE_OUTCOME_KEYWORDS = {
    "death", "mortality", "survival", "expire", "los", "length of stay",
    "ventilat", "vasopressor", "dialysis", "icu", "discharge", "readmission",
}

#: a plausible upper horizon (hours) for linked ICU/hospital data (~1 year).
MAX_EMULABLE_HORIZON_H = 365 * 24.0

DATASETS = ("MIMIC-IV", "eICU-CRD")


class EmulabilityScore(BaseModel):
    nct_id: str
    dataset: str
    score: float = Field(..., ge=0.0, le=1.0)
    eligibility_ok: float
    exposure_ok: float
    outcome_ok: float
    is_sepsis: bool
    emulable: bool
    reasons: list[str] = Field(default_factory=list)


def _is_sepsis(spec: TargetTrialSpec) -> bool:
    hay = " ".join(filter(None, [
        (spec.condition or "").lower(),
        (spec.title or "").lower(),
        *[c.concept.lower() for c in spec.eligibility if c.concept],
    ]))
    return "sepsis" in hay or any(
        vocab.classify(c.concept) == "sepsis" for c in spec.eligibility if c.concept)


def _eligibility_component(spec: TargetTrialSpec, reasons: list[str]) -> float:
    if not spec.eligibility:
        reasons.append("no eligibility criteria specified (neutral)")
        return 0.5
    ok = sum(1 for c in spec.eligibility if c.event_type in EHR_EVENT_TYPES)
    frac = ok / len(spec.eligibility)
    if frac < 1.0:
        reasons.append(f"{len(spec.eligibility) - ok} eligibility criteria use "
                       "non-EHR event types")
    return frac


def _exposure_component(spec: TargetTrialSpec, reasons: list[str]) -> float:
    treatment = [a for a in spec.arms if not a.is_control]
    if not treatment:
        reasons.append("no treatment arm")
        return 0.0
    identifiable = [a for a in treatment if a.intervention_concepts]
    if not identifiable:
        reasons.append("treatment arm(s) have no identifiable intervention concept")
        return 0.0
    return len(identifiable) / len(treatment)


def _outcome_component(spec: TargetTrialSpec, reasons: list[str]) -> float:
    if not spec.outcomes:
        reasons.append("no outcome specified")
        return 0.0
    best = 0.0
    for o in spec.outcomes:
        c = (o.concept or o.name or "").lower()
        # measurable iff a known endpoint keyword (death/LOS/...) OR a direct EHR
        # signal (lab/measurement). A generic OUTCOME concept (qol, satisfaction)
        # is NOT auto-measurable — only mortality-like keywords are.
        measurable = any(k in c for k in MEASURABLE_OUTCOME_KEYWORDS) \
            or o.event_type in (EventType.LAB, EventType.MEASUREMENT)
        horizon_ok = o.horizon_hours is None or o.horizon_hours <= MAX_EMULABLE_HORIZON_H
        if measurable and horizon_ok:
            best = max(best, 1.0)
        elif measurable and not horizon_ok:
            best = max(best, 0.4)
            reasons.append(f"outcome '{o.name}' horizon exceeds linked-data window")
        else:
            reasons.append(f"outcome '{o.name}' not reliably EHR-measurable")
    return best


def score_spec(spec: TargetTrialSpec, dataset: str, *, threshold: float = 0.5) -> EmulabilityScore:
    """Score one trial's emulability in one dataset. Never raises; always returns
    a row (low-emulability is flagged via `emulable=False` + reasons, not dropped)."""
    reasons: list[str] = []
    elig = _eligibility_component(spec, reasons)
    expo = _exposure_component(spec, reasons)
    outc = _outcome_component(spec, reasons)
    score = round(0.4 * elig + 0.3 * expo + 0.3 * outc, 4)
    emulable = score >= threshold and expo > 0 and outc > 0
    if not emulable and not reasons:
        reasons.append(f"score {score} below threshold {threshold}")
    return EmulabilityScore(
        nct_id=spec.nct_id, dataset=dataset, score=score,
        eligibility_ok=round(elig, 4), exposure_ok=round(expo, 4), outcome_ok=round(outc, 4),
        is_sepsis=_is_sepsis(spec), emulable=emulable, reasons=reasons,
    )


def build_catalog(specs: list[TargetTrialSpec], *, datasets: tuple[str, ...] = DATASETS,
                  threshold: float = 0.5) -> dict:
    """Score every (trial, dataset) and return a sortable catalog + a summary.

    Sort: sepsis first, then emulable, then score desc. NOTHING is dropped — every
    trial appears with its score + reasons (jpic: never silently cap), and the
    summary logs the drop-reasons distribution for the low-emulability tail.
    """
    rows: list[EmulabilityScore] = []
    for spec in specs:
        for ds in datasets:
            rows.append(score_spec(spec, ds, threshold=threshold))
    rows.sort(key=lambda r: (not r.is_sepsis, not r.emulable, -r.score, r.nct_id, r.dataset))

    emulable = [r for r in rows if r.emulable]
    sepsis = [r for r in rows if r.is_sepsis]
    dropped = [r for r in rows if not r.emulable]
    reason_counts: dict[str, int] = {}
    for r in dropped:
        for reason in r.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "catalog": [r.model_dump() for r in rows],
        "summary": {
            "n_trials": len(specs),
            "n_rows": len(rows),                  # trials x datasets
            "n_emulable": len(emulable),
            "n_sepsis_rows": len(sepsis),
            "n_emulable_sepsis": sum(1 for r in emulable if r.is_sepsis),
            "n_not_emulable": len(dropped),
            "not_emulable_reasons": dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
            "datasets": list(datasets),
            "threshold": threshold,
        },
    }


__all__ = ["EmulabilityScore", "score_spec", "build_catalog", "DATASETS", "EHR_EVENT_TYPES"]
