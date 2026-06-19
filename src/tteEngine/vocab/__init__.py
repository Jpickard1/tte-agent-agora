"""vocab: concept / unit normalization layer (#5, worker1).

Maps raw EVENT_NAME (+ value/unit) from the canonical 5-col stream to cross-DB
concept ids and normalized value+unit, producing the ``NormalizedEvent`` sidecar
(``contracts.events``). Keeps STRUCTURE (the 5 columns) decoupled from SEMANTICS
(this layer): adapters emit raw events; the analysis layer reads the sidecar for
typed value+unit (threshold eligibility/outcomes).

Two directions, both used by the adapters (#6/#7/#8) and the intelligence (#3):
  * ``resolve(concept)``  -> the set of source codes/names that mean that concept
    (what the per-DB adapters match on),
  * ``classify(name)``    -> the concept id a raw code/name belongs to.

Self-contained per jpic's directive: the seed dictionaries below are COPIED into
this repo (not imported), adapted from EHR-DE/MIMIC-IV/clinical_constants.py
(sepsis ICD-9/10, vasopressors) — credited here, extensible in-repo.
"""
from __future__ import annotations

import json
import re

from tteEngine.contracts.events import Event, EventType, NormalizedEvent

# --- seed concept dictionaries (copied + adapted from EHR-DE clinical_constants;
#     this repo owns its copy). concept_id -> set of source codes/names. --------

SEPSIS_ICD10 = {
    "A40", "A400", "A401", "A403", "A408", "A409", "A41", "A410", "A411", "A412",
    "A413", "A414", "A415", "A4150", "A4151", "A4152", "A4153", "A4159", "A418",
    "A4181", "A4189", "A419", "A499", "R6520", "R6521", "R7881",
}
SEPSIS_ICD9 = {
    "0380", "03810", "03811", "03812", "03819", "0382", "0383", "03840", "03841",
    "03842", "03843", "03844", "03849", "0388", "0389", "78552", "99591", "99592",
}
VASOPRESSORS = {
    "norepinephrine", "levophed", "noradrenaline", "epinephrine", "adrenaline",
    "dopamine", "vasopressin", "pitressin", "phenylephrine", "neosynephrine",
    "dobutamine", "milrinone",
}
CORTICOSTEROIDS = {
    "hydrocortisone", "methylprednisolone", "prednisone", "prednisolone",
    "dexamethasone", "fludrocortisone",
}

#: concept_id -> source codes/names. Extend this dict in-repo as trials need it.
CONCEPT_CODES: dict[str, set[str]] = {
    "sepsis": SEPSIS_ICD10 | SEPSIS_ICD9,
    "vasopressor": VASOPRESSORS,
    "corticosteroid": CORTICOSTEROIDS,
}

#: reverse index: a source code/name (lowercased) -> concept_id.
_CODE_TO_CONCEPT: dict[str, str] = {
    code.lower(): concept
    for concept, codes in CONCEPT_CODES.items()
    for code in codes
}

#: simple unit canonicalization (extensible). lowercased source unit -> canonical.
_UNIT_CANON = {
    "mg/dl": "mg/dL", "mmol/l": "mmol/L", "meq/l": "mEq/L", "g/dl": "g/dL",
    "bpm": "bpm", "mmhg": "mmHg", "%": "%", "k/ul": "K/uL", "/min": "/min",
}

# unit embedded in a name like "Creatinine (mg/dL)"
_NAME_UNIT_RE = re.compile(r"\(([^)]+)\)\s*$")


def register_concept(concept_id: str, codes: set[str]) -> None:
    """Add/extend a concept in-repo (keeps the reverse index in sync)."""
    CONCEPT_CODES.setdefault(concept_id, set()).update(codes)
    for c in codes:
        _CODE_TO_CONCEPT[c.lower()] = concept_id


def resolve(concept: str) -> set[str]:
    """concept id -> the set of source codes/names that mean it. Unknown concepts
    fall back to {concept} so adapters can still match a literal code/name."""
    return CONCEPT_CODES.get(concept, {concept})


def classify(event_name: str) -> str | None:
    """raw code/name -> its concept id, or None if unmapped."""
    return _CODE_TO_CONCEPT.get((event_name or "").strip().lower())


def normalize_value(event_value: str, event_name: str = "") -> tuple[float | None, str | None, str | None]:
    """(value_num, unit, value_text) from a raw EVENT_VALUE (str or JSON) +
    EVENT_NAME. Numeric scalars -> value_num; a unit in JSON or in the name
    ('X (mg/dL)') -> canonical unit; otherwise the raw string -> value_text."""
    raw = "" if event_value is None else str(event_value)
    unit: str | None = None

    # unit embedded in the name, e.g. "Creatinine (mg/dL)"
    m = _NAME_UNIT_RE.search(event_name or "")
    if m:
        unit = _UNIT_CANON.get(m.group(1).strip().lower(), m.group(1).strip())

    # JSON metadata value (meds/micro/inputevents carry dicts)
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for uk in ("unit", "uom", "valueuom", "amountuom", "rateuom"):
                if obj.get(uk):
                    unit = _UNIT_CANON.get(str(obj[uk]).lower(), str(obj[uk]))
                    break
            for vk in ("value", "valuenum", "amount", "dose", "rate"):
                if vk in obj:
                    try:
                        return float(obj[vk]), unit, None
                    except (TypeError, ValueError):
                        pass
            return None, unit, raw

    # plain numeric scalar
    try:
        return float(raw), unit, None
    except ValueError:
        return None, unit, (raw or None)


def to_normalized_event(event: Event) -> NormalizedEvent:
    """Build the sidecar for one canonical Event: concept id + typed value/unit."""
    value_num, unit, value_text = normalize_value(event.event_value, event.event_name)
    return NormalizedEvent(
        trajectory_id=event.trajectory_id,
        timestamp=event.timestamp,
        event_type=event.event_type,
        concept_id=classify(event.event_name),
        value_num=value_num,
        unit=unit,
        value_text=value_text,
    )


__all__ = [
    "CONCEPT_CODES", "register_concept", "resolve", "classify",
    "normalize_value", "to_normalized_event",
]
