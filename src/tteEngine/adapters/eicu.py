"""eICU-CRD -> canonical 5-col adapter (#7, worker1).

Same ExtractionPlan -> canonical 5-col contract as the MIMIC adapter (#6): given
a plan, extract only the cohort + requested concepts from eICU-CRD and emit a
stream that passes ``common_format.validate_canonical``.

eICU difference handled here: eICU is de-identified, so event times are OFFSETS
(minutes from unit admission, offset 0), not wall-clock timestamps. We build a
canonical tz-aware TIMESTAMP as ``EPOCH + offset_minutes`` — per-stay relative,
ordered and sub-day, which is what landmark/immortal-time logic needs (absolute
wall-clock is not recoverable in eICU, by design).

Self-contained per jpic: the eICU table/column map lives in this repo (TABLE_SPEC
below), no import from trialsim/EHR-DE. Table I/O is injectable (`tables`) so this
is unit-testable on synthetic fixtures; ``load_eicu_tables`` is the real loader.
"""
from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd

from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ExtractionPlan
from tteEngine.timing import effective_window, harmonize_timestamps

#: synthetic anchor for eICU offset->timestamp (times are relative-by-design).
EPOCH = pd.Timestamp("2000-01-01 00:00", tz="UTC")

#: eICU-CRD table/column map for the NAME/VALUE long tables (generic loop).
#: `offset` is minutes from unit admission. MEASUREMENT (wide vitals tables) and
#: OUTCOME (patient discharge status) are handled by dedicated helpers below, the
#: way the MIMIC adapter special-cases deathtime — so they are NOT in TABLE_SPEC.
TABLE_SPEC: dict[EventType, dict[str, str]] = {
    EventType.DIAGNOSIS: {"table": "diagnosis", "offset": "diagnosisoffset",
                          "name": "icd9code", "value": "diagnosisstring"},
    EventType.LAB: {"table": "lab", "offset": "labresultoffset",
                    "name": "labname", "value": "labresult"},
    EventType.MEDICATION: {"table": "medication", "offset": "drugstartoffset",
                           "name": "drugname", "value": "dosage"},
}

#: eICU vitals are WIDE (one row, many vital columns) in vitalPeriodic/Aperiodic.
#: This maps each source column -> a canonical vital concept name; the adapter
#: MELTS the requested columns into 5-col MEASUREMENT events.
VITAL_TABLES: tuple[str, ...] = ("vitalperiodic", "vitalaperiodic")
VITAL_OFFSET = "observationoffset"
VITAL_COLUMNS: dict[str, str] = {
    "heartrate": "heart_rate", "respiration": "resp_rate", "sao2": "spo2",
    "temperature": "temperature", "cvp": "cvp",
    "systemicsystolic": "sbp", "systemicdiastolic": "dbp", "systemicmean": "map",
    "noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp", "noninvasivemean": "map",
}

#: eICU mortality lives in patient.{unit,hospital}dischargestatus == 'Expired'.
#: Hospital-level is preferred (captures in-hospital deaths beyond ICU discharge).
_MORTALITY_SOURCES: tuple[tuple[str, str], ...] = (
    ("hospitaldischargestatus", "hospitaldischargeoffset"),
    ("unitdischargestatus", "unitdischargeoffset"),
)
_MORTALITY_KEYWORDS = ("death", "mortal", "expire", "surviv", "died", "fatal")

Resolver = Callable[[str], set[str]]
_STAY = "patientunitstayid"


def _identity_resolver(concept: str) -> set[str]:
    return {concept}


def _empty_canonical() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in CANONICAL_COLUMNS})
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    return df


def _ts(offset_min: pd.Series) -> pd.Series:
    return EPOCH + pd.to_timedelta(pd.to_numeric(offset_min, errors="coerce"), unit="m")


def _icd_match(series: "pd.Series", codes: set[str]) -> "pd.Series":
    """Boolean mask for eICU icd9code, which packs SEVERAL codes per field
    ('995.92, A41.9'). A row matches if any requested code is one of its
    comma/semicolon-separated tokens. Single-code fields work as before."""
    codeset = set(codes)

    def hit(val: str) -> bool:
        toks = {t.strip() for t in str(val).replace(";", ",").split(",") if t.strip()}
        return bool(toks & codeset)

    return series.astype(str).map(hit)


def _cohort_stays(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame],
                  resolve: Resolver) -> set[int]:
    """In-cohort stays = those with a diagnosis matching any cohort-filter concept.
    Empty filter -> all stays in the `patient` table."""
    if not plan.cohort_filter_concepts:
        pt = tables.get("patient")
        return set(pt[_STAY].astype("int64")) if pt is not None else set()
    codes: set[str] = set()
    for c in plan.cohort_filter_concepts:
        codes |= resolve(c)
    dx = tables.get("diagnosis")
    if dx is None or dx.empty:
        return set()
    hit = dx[_icd_match(dx["icd9code"], codes)]
    return set(hit[_STAY].astype("int64"))


def _extract_vitals(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame],
                    stays: set[int], lo_min: float, hi_min: float,
                    resolve: Resolver) -> list[pd.DataFrame]:
    """MELT the requested MEASUREMENT (vital) columns from vitalPeriodic/Aperiodic
    into 5-col events. A request matches a source column when the column's canonical
    vital name (VITAL_COLUMNS) is in the request's resolved concept set."""
    wanted: set[str] = set()
    for req in plan.concepts:
        if req.event_type == EventType.MEASUREMENT:
            wanted |= resolve(req.concept) | {req.concept}
    if not wanted:
        return []
    parts: list[pd.DataFrame] = []
    for tname in VITAL_TABLES:
        src = tables.get(tname)
        if src is None or src.empty or VITAL_OFFSET not in src.columns:
            continue
        in_cohort = src[src[_STAY].astype("int64").isin(stays)].copy()
        if in_cohort.empty:
            continue
        off = pd.to_numeric(in_cohort[VITAL_OFFSET], errors="coerce")
        in_win = in_cohort[(off >= lo_min) & (off <= hi_min)]
        if in_win.empty:
            continue
        for col, vital in VITAL_COLUMNS.items():
            if col not in in_win.columns or vital not in wanted:
                continue
            sub = in_win[[_STAY, VITAL_OFFSET, col]].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            sub = sub.dropna(subset=[col])
            if sub.empty:
                continue
            parts.append(pd.DataFrame({
                "TRAJECTORY_ID": sub[_STAY].astype("int64"),
                "TIMESTAMP": _ts(sub[VITAL_OFFSET]),
                "EVENT_TYPE": EventType.MEASUREMENT.value,
                "EVENT_NAME": vital,
                "EVENT_VALUE": sub[col].astype(str),
            }))
    return parts


def _extract_mortality(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame],
                       stays: set[int]) -> list[pd.DataFrame]:
    """Emit OUTCOME (death) events for stays whose patient discharge status is
    'Expired'. Like MIMIC's deathtime: NOT window-clipped (post-baseline), one
    event per mortality-like outcome concept, EVENT_NAME=the concept. Uses the
    first discharge source (hospital preferred) that has data."""
    death_reqs = [r for r in plan.concepts
                  if r.event_type == EventType.OUTCOME
                  and any(k in (r.concept or "").lower() for k in _MORTALITY_KEYWORDS)]
    if not death_reqs:
        return []
    pt = tables.get("patient")
    if pt is None or pt.empty:
        return []
    pt = pt[pt[_STAY].astype("int64").isin(stays)]
    if pt.empty:
        return []
    for status_col, offset_col in _MORTALITY_SOURCES:
        if status_col not in pt.columns:
            continue
        dead = pt[pt[status_col].astype(str).str.strip().str.lower() == "expired"].copy()
        if dead.empty:
            continue
        off = (pd.to_numeric(dead[offset_col], errors="coerce")
               if offset_col in dead.columns else pd.Series(0, index=dead.index))
        dead = dead.assign(_off=off).dropna(subset=["_off"])
        if dead.empty:
            continue
        parts: list[pd.DataFrame] = []
        for req in death_reqs:
            parts.append(pd.DataFrame({
                "TRAJECTORY_ID": dead[_STAY].astype("int64"),
                "TIMESTAMP": _ts(dead["_off"]),
                "EVENT_TYPE": EventType.OUTCOME.value,
                "EVENT_NAME": req.concept,
                "EVENT_VALUE": "1",
            }))
        return parts  # first source with data wins
    return []


def extract(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame], *,
            resolve: Resolver | None = None, drug_matcher: dict | None = None) -> pd.DataFrame:
    """Build the canonical 5-col stream for `plan` from eICU `tables` (patient,
    diagnosis, lab, medication, vitalperiodic/vitalaperiodic). Covers DIAGNOSIS/
    LAB/MEDICATION (generic loop), MEASUREMENT (melted vitals) and OUTCOME
    (mortality from patient discharge status). When `drug_matcher` is supplied,
    MEDICATION is matched BY CODE (drughiclseqno) + emitted as the arm concept (#131
    seam A). Returns a validate_canonical df."""
    resolve = resolve or _identity_resolver
    stays = _cohort_stays(plan, tables, resolve)
    if not stays:
        return _empty_canonical()
    lo, hi = effective_window(plan)
    lo_min, hi_min = lo * 60.0, hi * 60.0

    parts: list[pd.DataFrame] = []
    for req in plan.concepts:
        spec = TABLE_SPEC.get(req.event_type)
        if spec is None:
            continue
        if req.event_type == EventType.MEDICATION and drug_matcher:
            continue   # #131 seam A: matched by code below, not by name
        src = tables.get(spec["table"])
        if src is None or src.empty:
            continue
        codes = resolve(req.concept)
        name_match = (_icd_match(src[spec["name"]], codes)        # multi-code icd9 field
                      if req.event_type == EventType.DIAGNOSIS
                      else src[spec["name"]].astype(str).isin(codes))
        rows = src[src[_STAY].astype("int64").isin(stays) & name_match].copy()
        if rows.empty:
            continue
        off = pd.to_numeric(rows[spec["offset"]], errors="coerce")
        rows = rows[(off >= lo_min) & (off <= hi_min)]
        if rows.empty:
            continue
        out = pd.DataFrame({
            "TRAJECTORY_ID": rows[_STAY].astype("int64"),
            "TIMESTAMP": _ts(rows[spec["offset"]]),
            "EVENT_TYPE": req.event_type.value,
            "EVENT_NAME": rows[spec["name"]].astype(str),
            "EVENT_VALUE": rows[spec["value"]].astype(str),
        })
        parts.append(out)

    # MEASUREMENT (wide vitals) + OUTCOME (mortality) via dedicated extractors
    parts.extend(_extract_vitals(plan, tables, stays, lo_min, hi_min, resolve))
    parts.extend(_extract_mortality(plan, tables, stays))

    # #131 seam A: code-based MEDICATION (drughiclseqno -> arm concept + JSON value)
    med = tables.get("medication")
    if drug_matcher and med is not None and not med.empty:
        from tteEngine.matching import assign_med_concepts, med_event_value
        rows = med[med[_STAY].astype("int64").isin(stays)].copy()
        off = pd.to_numeric(rows.get("drugstartoffset"), errors="coerce")
        rows = rows[(off >= lo_min) & (off <= hi_min)]
        if not rows.empty:
            concept, code, method = assign_med_concepts(rows, ("drughiclseqno",), drug_matcher)
            rows = rows.assign(_concept=concept, _code=code, _method=method)
            rows = rows[rows["_concept"].notna()]
            if not rows.empty:
                n = len(rows)
                names = rows["drugname"].tolist() if "drugname" in rows.columns else [None] * n
                doses = rows["dosage"].tolist() if "dosage" in rows.columns else [None] * n
                parts.append(pd.DataFrame({
                    "TRAJECTORY_ID": rows[_STAY].astype("int64"),
                    "TIMESTAMP": _ts(rows["drugstartoffset"]),
                    "EVENT_TYPE": EventType.MEDICATION.value,
                    "EVENT_NAME": rows["_concept"].astype(str),
                    "EVENT_VALUE": [med_event_value(raw_name=nm, code=cd, method=mt, dose=ds,
                                                    source_table="medication")
                                   for nm, cd, mt, ds in zip(names, rows["_code"], rows["_method"], doses)],
                }))

    if not parts:
        return _empty_canonical()
    df = pd.concat(parts, ignore_index=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    df = df.sort_values(["TRAJECTORY_ID", "TIMESTAMP"]).reset_index(drop=True)
    df = harmonize_timestamps(df, getattr(plan, "timing", None))  # #31 common precision grid
    return validate_canonical(df[list(CANONICAL_COLUMNS)])


def load_eicu_tables(eicu_dir: str, needed: list[str]) -> dict[str, pd.DataFrame]:
    """Real-data loader: read the needed eICU-CRD CSVs from `eicu_dir`. Thin +
    separate from `extract` so the adapter stays unit-testable without real data."""
    import pathlib
    base = pathlib.Path(eicu_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in needed:
        for cand in (base / f"{name}.csv.gz", base / f"{name}.csv"):
            if cand.exists():
                out[name] = pd.read_csv(cand, compression="infer")
                break
    return out
