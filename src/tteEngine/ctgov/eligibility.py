"""Rigorous eligibility (#28, worker1): ctgov free-text inclusion/exclusion
criteria -> EXECUTABLE predicates over the canonical 5-col stream.

#2's `study_to_spec` captures only the STRUCTURED demographics (age/sex bounds).
The clinically meaningful eligibility lives in the free-text
``eligibilityModule.eligibilityCriteria`` blob:

    Inclusion Criteria:
    * Adults >= 18 years with suspected or confirmed infection
    * Serum lactate > 2 mmol/L
    Exclusion Criteria:
    * Pregnancy
    * Receiving renal replacement therapy

This module turns those bullets into typed ``EligibilityCriterion`` predicates
(concept + comparator + value + unit + include flag) that ``build_cohort`` (#9)
can execute against the event stream — concepts are emitted as vocab concept_ids
so the #59 resolver matches raw-coded adapter streams (ICD/code) too.

It is deliberately a DETERMINISTIC, rule-based v1 (a lexicon + comparator
grammar) — no LLM, fully testable, extend the lexicon in-repo as trials need it.
Crucially it NEVER silently drops a bullet: every criterion line is either parsed
to a predicate or recorded in ``unparsed`` with the raw text, and ``coverage``
reports the parsed fraction. That visibility is the point — it's how we see how
much of each protocol became executable (jpic's maximize-count goal) instead of
hiding the gap. Pure stdlib (re) — no pandas.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..contracts import Comparator, EligibilityCriterion, EventType

# --- comparator grammar: phrase -> Comparator. Ordered longest/compound FIRST so
# 'greater than or equal to' wins over 'greater than'. Symbols handled too. ---
_COMPARATORS: list[tuple[str, Comparator]] = [
    ("greater than or equal to", Comparator.GE),
    ("less than or equal to", Comparator.LE),
    ("at least", Comparator.GE),
    ("no less than", Comparator.GE),
    ("at most", Comparator.LE),
    ("no more than", Comparator.LE),
    ("up to", Comparator.LE),
    ("greater than", Comparator.GT),
    ("more than", Comparator.GT),
    ("less than", Comparator.LT),
    ("equal to", Comparator.EQ),
    ("equals", Comparator.EQ),
    ("above", Comparator.GT),
    ("over", Comparator.GT),
    ("exceeding", Comparator.GT),
    ("below", Comparator.LT),
    ("under", Comparator.LT),
    (">=", Comparator.GE), ("≥", Comparator.GE),
    ("<=", Comparator.LE), ("≤", Comparator.LE),
    (">", Comparator.GT), ("<", Comparator.LT), ("=", Comparator.EQ),
]

# a number + optional unit, e.g. '2 mmol/L', '18 years', '100', '300mmHg'
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*([a-zµ%][a-z0-9µ%/]*(?:/[a-z0-9]+)?)?", re.I)

_UNIT_CANON = {
    "mmol/l": "mmol/L", "mg/dl": "mg/dL", "meq/l": "mEq/L", "g/dl": "g/dL",
    "mmhg": "mmHg", "bpm": "bpm", "%": "%", "k/ul": "K/uL",
    "years": "years", "year": "years", "yrs": "years", "yr": "years",
}

# --- concept lexicon: surface phrase -> (vocab concept_id, EventType). Ordered
# longest-first (so 'septic shock' beats 'sepsis'). concept_ids align with the
# vocab layer (#5) where one exists; new ids are fine — build_cohort's resolver
# falls back to a literal match. Extend in-repo as trials need it. ---
_LEXICON: list[tuple[str, tuple[str, EventType]]] = [
    ("septic shock", ("sepsis", EventType.DIAGNOSIS)),
    ("severe sepsis", ("sepsis", EventType.DIAGNOSIS)),
    ("sepsis", ("sepsis", EventType.DIAGNOSIS)),
    ("suspected infection", ("infection", EventType.DIAGNOSIS)),
    ("confirmed infection", ("infection", EventType.DIAGNOSIS)),
    ("infection", ("infection", EventType.DIAGNOSIS)),
    ("acute respiratory distress", ("ards", EventType.DIAGNOSIS)),
    ("acute kidney injury", ("aki", EventType.DIAGNOSIS)),
    ("serum lactate", ("lactate", EventType.LAB)),
    ("lactate", ("lactate", EventType.LAB)),
    ("serum creatinine", ("creatinine", EventType.LAB)),
    ("creatinine", ("creatinine", EventType.LAB)),
    ("bilirubin", ("bilirubin", EventType.LAB)),
    ("platelet count", ("platelet", EventType.LAB)),
    ("platelet", ("platelet", EventType.LAB)),
    ("white blood cell", ("wbc", EventType.LAB)),
    ("mean arterial pressure", ("map", EventType.MEASUREMENT)),
    ("systolic blood pressure", ("sbp", EventType.MEASUREMENT)),
    ("heart rate", ("heart_rate", EventType.MEASUREMENT)),
    ("respiratory rate", ("resp_rate", EventType.MEASUREMENT)),
    ("temperature", ("temperature", EventType.MEASUREMENT)),
    ("mechanically ventilated", ("mechanical_ventilation", EventType.PROCEDURE)),
    ("mechanical ventilation", ("mechanical_ventilation", EventType.PROCEDURE)),
    ("renal replacement", ("dialysis", EventType.PROCEDURE)),
    ("dialysis", ("dialysis", EventType.PROCEDURE)),
    ("norepinephrine", ("vasopressor", EventType.MEDICATION)),
    ("vasopressor", ("vasopressor", EventType.MEDICATION)),
    ("corticosteroid", ("corticosteroid", EventType.MEDICATION)),
    ("pregnant", ("pregnancy", EventType.DIAGNOSIS)),
    ("pregnancy", ("pregnancy", EventType.DIAGNOSIS)),
    ("age", ("age", EventType.DEMOGRAPHIC)),
]

_INCLUSION_HDR = re.compile(r"inclusion\s+criteria", re.I)
_EXCLUSION_HDR = re.compile(r"exclusion\s+criteria", re.I)
_BULLET = re.compile(r"^\s*(?:[-*•·o]|\d+[.)])\s+", re.M)


@dataclass
class EligibilityParse:
    """Result of parsing one trial's eligibility text. ``unparsed`` keeps the raw
    bullets we could not turn into predicates — nothing is silently dropped."""

    criteria: list[EligibilityCriterion] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    n_bullets: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of bullets turned into executable predicates (0..1)."""
        return 0.0 if self.n_bullets == 0 else round(len(self.criteria) / self.n_bullets, 4)


def _canon_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return _UNIT_CANON.get(unit.strip().lower(), unit.strip())


_NUMERIC_CMP = {Comparator.GT, Comparator.GE, Comparator.LT, Comparator.LE, Comparator.EQ}
#: concepts a numeric threshold can sensibly apply to (a labs/vitals/age value);
#: a DIAGNOSIS/MEDICATION/PROCEDURE concept is presence-only (EXISTS).
_NUMERIC_EVENT_TYPES = {EventType.LAB, EventType.MEASUREMENT, EventType.DEMOGRAPHIC}


def _find_comparator(text: str) -> tuple[Comparator, int] | None:
    """First comparator phrase/symbol present (longest/compound matched first),
    with the index just past it (so the number is searched AFTER the operator)."""
    low = text.lower()
    best: tuple[int, Comparator, int] | None = None
    for phrase, cmp in _COMPARATORS:
        i = low.find(phrase)
        if i != -1 and (best is None or i < best[0]):
            best = (i, cmp, i + len(phrase))
    return (best[1], best[2]) if best else None


def _find_concept(text: str, classify: Callable[[str], str | None] | None) -> tuple[str, EventType] | None:
    """Map a bullet to (concept_id, event_type) via the lexicon (longest match),
    then fall back to the vocab classifier on individual tokens (catches raw codes)."""
    low = text.lower()
    for phrase, hit in _LEXICON:
        if phrase in low:
            return hit
    if classify is not None:
        for tok in re.findall(r"[a-z0-9.]+", low):
            cid = classify(tok)
            if cid:
                # event type unknown from a bare code -> diagnosis is the safe default
                return (cid, EventType.DIAGNOSIS)
    return None


def _split_sections(text: str) -> list[tuple[str, bool]]:
    """Split the blob into (segment_text, include) parts on inclusion/exclusion
    headers. Default (no header) is inclusion."""
    incl_m = _INCLUSION_HDR.search(text)
    excl_m = _EXCLUSION_HDR.search(text)
    if not excl_m:
        return [(text, True)]
    start = incl_m.end() if incl_m else 0
    return [(text[start:excl_m.start()], True), (text[excl_m.end():], False)]


def _bullets(segment: str) -> list[str]:
    """Split a segment into bullet lines. Handles -/*/•/numbered markers and, if a
    blob has no markers, falls back to non-empty lines."""
    if _BULLET.search(segment):
        parts = _BULLET.split(segment)
    else:
        parts = re.split(r"[\r\n]+", segment)
    # require some alphanumeric content -> drops header-colon leftovers (':') etc.
    return [p.strip() for p in parts if p and re.search(r"[a-z0-9]", p, re.I)]


def parse_eligibility_text(
    text: str | None, *, classify: Callable[[str], str | None] | None = None,
    window_hours: tuple[float, float] | None = None,
) -> EligibilityParse:
    """Parse a free-text eligibility blob into executable predicates.

    `classify` (optional): a vocab resolver (e.g. ``vocab.classify``) used as a
    fallback to recognise raw codes embedded in the text. `window_hours` stamps a
    relative-to-time-zero window on numeric (lab/measurement) predicates.
    """
    out = EligibilityParse()
    if not text or not text.strip():
        return out

    for segment, include in _split_sections(text):
        for bullet in _bullets(segment):
            out.n_bullets += 1
            concept = _find_concept(bullet, classify)
            if concept is None:
                out.unparsed.append(bullet)
                continue
            cid, et = concept
            cm = _find_comparator(bullet)
            crit: EligibilityCriterion | None = None
            if cm and cm[0] in _NUMERIC_CMP and et in _NUMERIC_EVENT_TYPES:
                cmp, pos = cm
                m = _NUM_UNIT.search(bullet, pos)        # number AFTER the operator
                if m:
                    value = float(m.group(1))
                    unit = _canon_unit(m.group(2))
                    win = window_hours if et in (EventType.LAB, EventType.MEASUREMENT) else None
                    crit = EligibilityCriterion(concept=cid, event_type=et, comparator=cmp,
                                                value=value, unit=unit, window_hours=win,
                                                include=include)
            if crit is None:
                # a recognised concept but no numeric threshold -> presence predicate
                crit = EligibilityCriterion(concept=cid, event_type=et,
                                            comparator=Comparator.EXISTS, include=include)
            out.criteria.append(crit)
    return out


def parse_eligibility(
    study: dict, *, classify: Callable[[str], str | None] | None = None,
    window_hours: tuple[float, float] | None = None,
) -> EligibilityParse:
    """Parse a ctgov study dict's eligibility free-text (#1 reader output)."""
    text = (study.get("protocolSection", {})
                 .get("eligibilityModule", {})
                 .get("eligibilityCriteria"))
    return parse_eligibility_text(text, classify=classify, window_hours=window_hours)


def _key(c: EligibilityCriterion) -> tuple:
    return (c.concept, c.event_type, c.comparator, c.value, c.include)


def enrich_spec_eligibility(
    spec, study: dict, *, classify: Callable[[str], str | None] | None = None,
    window_hours: tuple[float, float] | None = None,
):
    """Merge free-text eligibility predicates into ``spec.eligibility`` in place,
    deduped against what's already there (e.g. #2's structured demographics).

    The integration seam for #2's ``study_to_spec`` (probe's file): call this after
    parsing to upgrade a demographics-only spec into one with executable clinical
    predicates. Returns the same spec (and stamps ``spec`` unchanged if no text).
    """
    parse = parse_eligibility(study, classify=classify, window_hours=window_hours)
    existing = {_key(c) for c in spec.eligibility}
    for c in parse.criteria:
        if _key(c) not in existing:
            spec.eligibility.append(c)
            existing.add(_key(c))
    return spec


__all__ = [
    "EligibilityParse", "parse_eligibility", "parse_eligibility_text",
    "enrich_spec_eligibility",
]
