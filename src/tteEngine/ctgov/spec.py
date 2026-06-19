"""ctgov study -> TargetTrialSpec (#2, probe / lane:analysis).

Parse a cached ClinicalTrials.gov study (from the #1 reader) into the typed
``TargetTrialSpec`` (PICO-T + estimand): condition, treatment/comparator arms,
primary/secondary outcomes, and structured demographic eligibility. Deeper
free-text eligibility -> typed predicates is the #3 "intelligence" step; here we
capture the protocol skeleton + demographics deterministically.

Reuses the field layout from trialsim's ctgov schema; emits emulaTTE-style
contracts so the rest of the pipeline is dataset-agnostic.
"""
from __future__ import annotations

import re

from ..contracts import (
    Arm,
    Comparator,
    EligibilityCriterion,
    Estimand,
    EventType,
    OutcomeSpec,
    TargetTrialSpec,
    TimeZeroRule,
)

# arm labels / types that denote the comparator (control) arm
_CONTROL_RE = re.compile(r"placebo|standard|usual care|control|sham|no intervention", re.I)
_CONTROL_TYPES = {"PLACEBO_COMPARATOR", "NO_INTERVENTION", "SHAM_COMPARATOR"}

_AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day|hour)s?", re.I)
_TIMEFRAME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(year|month|week|day|hour)s?", re.I)
_UNIT_HOURS = {"year": 8760.0, "month": 730.0, "week": 168.0, "day": 24.0, "hour": 1.0}


def _ps(study: dict) -> dict:
    return study.get("protocolSection", {})


def _age_to_years(text: str | None) -> float | None:
    """'18 Years' / '6 Months' -> years (float). None for 'N/A'/blank/unparseable."""
    if not text:
        return None
    m = _AGE_RE.search(text)
    if not m:
        return None
    n, unit = float(m.group(1)), m.group(2).lower()
    return n * (_UNIT_HOURS[unit] / _UNIT_HOURS["year"])


def _timeframe_to_hours(text: str | None) -> float | None:
    """Best-effort outcome time-frame -> hours ('28 days' -> 672). None if unparseable."""
    if not text:
        return None
    m = _TIMEFRAME_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) * _UNIT_HOURS[m.group(2).lower()]


def _arms(study: dict) -> list[Arm]:
    mod = _ps(study).get("armsInterventionsModule", {})
    out: list[Arm] = []
    for g in mod.get("armGroups", []) or []:
        label = (g.get("label") or "").strip()
        if not label:
            continue
        is_control = (g.get("type") in _CONTROL_TYPES) or bool(_CONTROL_RE.search(label))
        out.append(
            Arm(
                name=label,
                is_control=is_control,
                intervention_concepts=[s for s in (g.get("interventionNames") or []) if s],
            )
        )
    return out


def _outcomes(study: dict) -> list[OutcomeSpec]:
    mod = _ps(study).get("outcomesModule", {})
    out: list[OutcomeSpec] = []
    for key in ("primaryOutcomes", "secondaryOutcomes"):
        for o in mod.get(key, []) or []:
            measure = (o.get("measure") or "").strip()
            if not measure:
                continue
            out.append(
                OutcomeSpec(
                    name=measure,
                    event_type=EventType.OUTCOME,
                    horizon_hours=_timeframe_to_hours(o.get("timeFrame")),
                )
            )
    return out


def _eligibility(study: dict) -> list[EligibilityCriterion]:
    """Structured DEMOGRAPHIC predicates (age bounds, sex). Free-text inclusion/
    exclusion -> typed predicates is the #3 intelligence step."""
    elig = _ps(study).get("eligibilityModule", {})
    crit: list[EligibilityCriterion] = []
    lo = _age_to_years(elig.get("minimumAge"))
    hi = _age_to_years(elig.get("maximumAge"))
    if lo is not None:
        crit.append(EligibilityCriterion(concept="age", event_type=EventType.DEMOGRAPHIC,
                                         comparator=Comparator.GE, value=lo, unit="years"))
    if hi is not None:
        crit.append(EligibilityCriterion(concept="age", event_type=EventType.DEMOGRAPHIC,
                                         comparator=Comparator.LE, value=hi, unit="years"))
    sex = (elig.get("sex") or "ALL").upper()
    if sex in ("FEMALE", "MALE"):
        crit.append(EligibilityCriterion(concept="sex", event_type=EventType.DEMOGRAPHIC,
                                         comparator=Comparator.EQ, value=sex))
    return crit


def study_to_spec(study: dict) -> TargetTrialSpec:
    """Parse a ctgov study dict (from the #1 reader) into a TargetTrialSpec."""
    ps = _ps(study)
    ident = ps.get("identificationModule", {})
    conds = ps.get("conditionsModule", {}).get("conditions", []) or []
    return TargetTrialSpec(
        nct_id=ident.get("nctId", ""),
        title=ident.get("briefTitle") or ident.get("officialTitle"),
        condition=conds[0] if conds else None,
        eligibility=_eligibility(study),
        arms=_arms(study),
        outcomes=_outcomes(study),
        time_zero=TimeZeroRule(),
        estimand=Estimand.INTENTION_TO_TREAT,
    )
