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

#: synthetic anchor for eICU offset->timestamp (times are relative-by-design).
EPOCH = pd.Timestamp("2000-01-01 00:00", tz="UTC")

#: eICU-CRD table/column map. `offset` is minutes from unit admission.
TABLE_SPEC: dict[EventType, dict[str, str]] = {
    EventType.DIAGNOSIS: {"table": "diagnosis", "offset": "diagnosisoffset",
                          "name": "icd9code", "value": "diagnosisstring"},
    EventType.LAB: {"table": "lab", "offset": "labresultoffset",
                    "name": "labname", "value": "labresult"},
    EventType.MEDICATION: {"table": "medication", "offset": "drugstartoffset",
                           "name": "drugname", "value": "dosage"},
}

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
    hit = dx[dx["icd9code"].astype(str).isin(codes)]
    return set(hit[_STAY].astype("int64"))


def extract(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame], *,
            resolve: Resolver | None = None) -> pd.DataFrame:
    """Build the canonical 5-col stream for `plan` from eICU `tables`
    (patient, diagnosis, lab, medication, ...). Returns a validate_canonical df."""
    resolve = resolve or _identity_resolver
    stays = _cohort_stays(plan, tables, resolve)
    if not stays:
        return _empty_canonical()
    lo, hi = plan.window_hours
    lo_min, hi_min = lo * 60.0, hi * 60.0

    parts: list[pd.DataFrame] = []
    for req in plan.concepts:
        spec = TABLE_SPEC.get(req.event_type)
        if spec is None:
            continue
        src = tables.get(spec["table"])
        if src is None or src.empty:
            continue
        codes = resolve(req.concept)
        rows = src[src[_STAY].astype("int64").isin(stays)
                   & src[spec["name"]].astype(str).isin(codes)].copy()
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

    if not parts:
        return _empty_canonical()
    df = pd.concat(parts, ignore_index=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    df = df.sort_values(["TRAJECTORY_ID", "TIMESTAMP"]).reset_index(drop=True)
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
