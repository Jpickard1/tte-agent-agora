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

import re

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
    # data-driven emulability (#122): set only when a #109 vocab index is supplied.
    condition_resolves: bool | None = None     # does the cohort concept hit >=1 real code here?
    n_condition_codes: int | None = None        # how many real diagnosis codes it resolves to


#: sepsis-family stems so 'septic shock' / 'severe sepsis' / 'septicaemia' all
#: flag (jpic's sepsis priority — matching only 'sepsis' drops most real RCTs,
#: which name the population 'septic shock'). 'septic' covers septic*/septicemia.
SEPSIS_STEMS = ("sepsis", "septic")


def _is_sepsis(spec: TargetTrialSpec) -> bool:
    hay = " ".join(filter(None, [
        (spec.condition or "").lower(),
        (spec.title or "").lower(),
        *[c.concept.lower() for c in spec.eligibility if c.concept],
    ]))
    return any(stem in hay for stem in SEPSIS_STEMS) or any(
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


#: ctgov intervention-type prefixes that are NOT drugs -> not emulable in ICU drug
#: data (the real run found TORAYMYXIN/Starling-SV device trials yield no med events).
_NONDRUG_TYPES = ("device", "procedure", "radiation", "behavioral", "diagnostic test",
                  "genetic", "other")
#: high-prevalence ICU "background" drugs — given to ~everyone (banana bag), so a
#: trial whose ONLY distinguishing intervention is one of these has ambiguous arms
#: (#122/#162-B: down-rank, since 'treated' ~= 'got routine care').
ROUTINE_DRUGS = {"thiamine", "multivitamin", "vitamin", "vitamin c", "ascorbic acid",
                 "folic acid", "folate", "magnesium", "normal saline", "saline",
                 "sodium chloride", "dextrose", "potassium chloride", "acetaminophen",
                 "heparin", "famotidine", "pantoprazole", "docusate", "senna"}


def _intervention_type(concept: str) -> str:
    m = re.match(r"\s*([a-z][a-z ]*?)\s*:", (concept or "").lower())
    return m.group(1).strip() if m else "drug"     # unprefixed -> assume drug


def _strip_type(concept: str) -> str:
    return re.sub(r"^\s*[a-z][a-z ]*?\s*:\s*", "", (concept or "").lower()).strip()


def _exposure_component(spec: TargetTrialSpec, reasons: list[str]) -> float:
    treatment = [a for a in spec.arms if not a.is_control]
    if not treatment:
        reasons.append("no treatment arm")
        return 0.0
    identifiable = [a for a in treatment if a.intervention_concepts]
    if not identifiable:
        reasons.append("treatment arm(s) have no identifiable intervention concept")
        return 0.0
    # #122: a DEVICE/non-drug intervention isn't emulable in ICU drug data
    concepts = [c for a in identifiable for c in a.intervention_concepts]
    drugs = [c for c in concepts if _intervention_type(c) not in _NONDRUG_TYPES]
    if not drugs:
        reasons.append("intervention is device/non-drug — not emulable in ICU drug data")
        return 0.0
    score = len(identifiable) / len(treatment)
    # #122/#162-B: distinguishing-drug — if EVERY drug component is routine/high-
    # prevalence, the arms are ambiguous ('treated' ~= routine care) -> down-rank.
    if all(_strip_type(c) in ROUTINE_DRUGS for c in drugs):
        reasons.append("intervention is routine high-prevalence drug(s) — arms likely "
                       "ambiguous (over-inclusion); not a distinguishing treatment")
        score *= 0.5
    return score


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


def _codes_for(index: dict, category: str, *terms: str) -> set[str]:
    """Pure concept->real-codes lookup over a #109 vocab-index DICT (no pandas, so
    triage stays import-light): codes whose name/code contains any term (ci)."""
    entries = (index or {}).get("categories", {}).get(category, [])
    out: set[str] = set()
    for term in terms:
        t = term.lower()
        out |= {e["code"] for e in entries if t in e["name"].lower() or t in e["code"].lower()}
    return out


def _resolve_condition_codes(spec: TargetTrialSpec, index: dict) -> tuple[set[str], list[str]]:
    """Real diagnosis codes the trial's cohort-defining concepts (condition + dx
    eligibility) resolve to in this dataset's index — full phrase first, else alpha
    tokens (so 'Sepsis-3' -> 'sepsis')."""
    concepts = [spec.condition] if spec.condition else []
    concepts += [c.concept for c in spec.eligibility
                 if c.event_type == EventType.DIAGNOSIS and c.concept]
    codes: set[str] = set()
    for c in concepts:
        got = _codes_for(index, "diagnosis", c.lower())
        if not got:
            toks = [t for t in re.split(r"[^a-z0-9]+", c.lower()) if len(t) >= 4]
            if toks:
                got = _codes_for(index, "diagnosis", *toks)
        codes |= got
    return codes, concepts


def score_spec(spec: TargetTrialSpec, dataset: str, *, threshold: float = 0.5,
               index: dict | None = None) -> EmulabilityScore:
    """Score one trial's emulability in one dataset. Never raises; always returns
    a row (low-emulability is flagged via `emulable=False` + reasons, not dropped).

    `index` (optional #109 per-dataset vocab-index dict): makes emulability
    DATA-DRIVEN (#122) — the cohort-defining concept must resolve to >=1 REAL
    diagnosis code in this dataset, else the trial is NOT emulable here (a trial
    can be structurally emulable yet have an empty cohort). Without an index,
    scoring is the structural heuristic as before (backward-compatible)."""
    reasons: list[str] = []
    elig = _eligibility_component(spec, reasons)
    expo = _exposure_component(spec, reasons)
    outc = _outcome_component(spec, reasons)
    score = round(0.4 * elig + 0.3 * expo + 0.3 * outc, 4)
    emulable = score >= threshold and expo > 0 and outc > 0

    condition_resolves = n_condition_codes = None
    if index is not None:
        codes, concepts = _resolve_condition_codes(spec, index)
        n_condition_codes = len(codes)
        if concepts:
            condition_resolves = n_condition_codes > 0
            if not condition_resolves:
                emulable = False
                reasons.append(f"cohort concept(s) {concepts} resolve to no diagnosis "
                               f"codes in {dataset} (empty cohort)")

    if not emulable and not reasons:
        reasons.append(f"score {score} below threshold {threshold}")
    return EmulabilityScore(
        nct_id=spec.nct_id, dataset=dataset, score=score,
        eligibility_ok=round(elig, 4), exposure_ok=round(expo, 4), outcome_ok=round(outc, 4),
        is_sepsis=_is_sepsis(spec), emulable=emulable, reasons=reasons,
        condition_resolves=condition_resolves, n_condition_codes=n_condition_codes,
    )


def build_catalog(specs: list[TargetTrialSpec], *, datasets: tuple[str, ...] = DATASETS,
                  threshold: float = 0.5, indexes: dict | None = None) -> dict:
    """Score every (trial, dataset) and return a sortable catalog + a summary.

    Sort: sepsis first, then emulable, then score desc. NOTHING is dropped — every
    trial appears with its score + reasons (jpic: never silently cap), and the
    summary logs the drop-reasons distribution for the low-emulability tail.

    `indexes` (optional {dataset: #109 vocab-index dict}) makes the catalog
    DATA-DRIVEN (#122): a trial counts as emulable in a dataset only if its cohort
    concept resolves to >=1 real code there, so the >=1k corpus count is TRUTHFUL
    (no emulable-but-empty-cohort trials).
    """
    rows: list[EmulabilityScore] = []
    for spec in specs:
        for ds in datasets:
            rows.append(score_spec(spec, ds, threshold=threshold,
                                   index=(indexes or {}).get(ds)))
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
            # data-driven (#122): how many rows were ruled out by zero real codes
            "n_condition_unresolved": sum(1 for r in rows if r.condition_resolves is False),
            "data_driven": indexes is not None,
            "datasets": list(datasets),
            "threshold": threshold,
        },
    }


from .runner import run_corpus_triage  # noqa: E402  (corpus runner, #35)

__all__ = ["EmulabilityScore", "score_spec", "build_catalog", "DATASETS",
           "EHR_EVENT_TYPES", "run_corpus_triage"]
