"""MIMIC-IV -> canonical 5-col adapter (#6, worker1).

Spec-driven: given an ``ExtractionPlan`` (#3), extract ONLY the cohort + the
concepts it requires from MIMIC-IV and emit the canonical 5-column event stream
(``contracts.events``), passing ``common_format.validate_canonical``.

This generalizes ``EHR-DE/MIMIC-IV/extraction_v1.py`` (which extracts the *whole*
cohort to the same 5-col format) by making cohort + feature selection
plan-driven. Table I/O is injectable (the ``tables`` mapping of DataFrames) so the
adapter is unit-testable on synthetic fixtures with no real MIMIC CSVs;
``load_mimic_tables`` is the real-data loader (reads the MIMIC-IV CSVs).

Concept resolution (raw code/name -> the concepts a plan asks for) is delegated to
the vocab layer (#5) via an injectable ``resolve`` callable, so #6 does not block
on #5: a plan concept resolves to the set of source codes/names to match.
"""
from __future__ import annotations

import json
from typing import Callable, Mapping

import pandas as pd

from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan

#: How each EventType maps to a MIMIC-IV source table + the columns we read.
#: A representative subset of EHR-DE/extraction_v1.py's ~20 sources; the same
#: pattern extends to the rest (chartevents/outputevents/microbiology/...).
TABLE_SPEC: dict[EventType, dict[str, str]] = {
    EventType.DIAGNOSIS: {"table": "diagnoses_icd", "time": "charttime",
                          "name": "icd_code", "value": "icd_code"},
    EventType.LAB: {"table": "labevents", "time": "charttime",
                    "name": "label", "value": "valuenum"},
    EventType.MEASUREMENT: {"table": "chartevents", "time": "charttime",
                            "name": "label", "value": "valuenum"},
    EventType.MEDICATION: {"table": "prescriptions", "time": "starttime",
                           "name": "drug", "value": "dose_val_rx"},
}

Resolver = Callable[[str], set[str]]


def _identity_resolver(concept: str) -> set[str]:
    """Fallback when no vocab layer (#5) is wired: match the concept literally."""
    return {concept}


def _empty_canonical() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in CANONICAL_COLUMNS})
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    return df


def _cohort_hadm_ids(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame],
                     resolve: Resolver) -> set[int]:
    """hadm_ids in-cohort = those with a diagnosis matching any cohort-filter
    concept. Empty filter -> every admission (from the `admissions` table)."""
    if not plan.cohort_filter_concepts:
        adm = tables.get("admissions")
        return set(adm["hadm_id"].astype("int64")) if adm is not None else set()
    codes: set[str] = set()
    for c in plan.cohort_filter_concepts:
        codes |= resolve(c)
    dx = tables.get("diagnoses_icd")
    if dx is None or dx.empty:
        return set()
    hit = dx[dx["icd_code"].astype(str).isin(codes)]
    return set(hit["hadm_id"].astype("int64"))


def _anchor_times(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame],
                  hadm_ids: set[int]) -> dict[int, pd.Timestamp]:
    """Per-admission anchor (time-zero) for the extraction window = admittime."""
    adm = tables.get("admissions")
    if adm is None:
        return {}
    a = adm[adm["hadm_id"].astype("int64").isin(hadm_ids)]
    return {int(h): pd.to_datetime(t, utc=True)
            for h, t in zip(a["hadm_id"], a["admittime"])}


def extract(plan: ExtractionPlan, tables: Mapping[str, pd.DataFrame], *,
            resolve: Resolver | None = None) -> pd.DataFrame:
    """Build the canonical 5-col stream for `plan` from the given MIMIC `tables`.

    `tables` keys are MIMIC table names (admissions, diagnoses_icd, labevents,
    chartevents, prescriptions, ...); only those needed by the plan are touched.
    Returns a DataFrame that passes ``validate_canonical``.
    """
    resolve = resolve or _identity_resolver
    hadm_ids = _cohort_hadm_ids(plan, tables, resolve)
    if not hadm_ids:
        return _empty_canonical()
    anchors = _anchor_times(plan, tables, hadm_ids)
    lo, hi = plan.window_hours

    parts: list[pd.DataFrame] = []
    for req in plan.concepts:
        spec = TABLE_SPEC.get(req.event_type)
        if spec is None:
            continue
        src = tables.get(spec["table"])
        if src is None or src.empty:
            continue
        codes = resolve(req.concept)
        rows = src[src["hadm_id"].astype("int64").isin(hadm_ids)
                   & src[spec["name"]].astype(str).isin(codes)].copy()
        if rows.empty:
            continue
        rows["__ts"] = pd.to_datetime(rows[spec["time"]], utc=True)
        # restrict to the extraction window around each admission's anchor
        if anchors:
            anc = rows["hadm_id"].astype("int64").map(anchors)
            keep = anc.notna() & (rows["__ts"] >= anc + pd.to_timedelta(lo, "h")) \
                & (rows["__ts"] <= anc + pd.to_timedelta(hi, "h"))
            rows = rows[keep]
        if rows.empty:
            continue
        out = pd.DataFrame({
            "TRAJECTORY_ID": rows["hadm_id"].astype("int64"),
            "TIMESTAMP": rows["__ts"],
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
    df = df[list(CANONICAL_COLUMNS)]
    return validate_canonical(df)


def load_mimic_tables(mimic_dir: str, needed: list[str]) -> dict[str, pd.DataFrame]:
    """Real-data loader: read the needed MIMIC-IV CSVs from `mimic_dir`. Kept
    thin + separate from `extract` so the adapter logic stays unit-testable
    without real data. (Full per-table column handling mirrors extraction_v1.py.)"""
    import pathlib
    base = pathlib.Path(mimic_dir)
    out: dict[str, pd.DataFrame] = {}
    for name in needed:
        for sub in ("hosp", "icu"):
            p = base / sub / f"{name}.csv.gz"
            if p.exists():
                out[name] = pd.read_csv(p, compression="gzip")
                break
    return out
