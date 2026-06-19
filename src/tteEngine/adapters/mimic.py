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
from tteEngine.timing import effective_window, harmonize_timestamps

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
                     resolve: Resolver, dx_matcher: dict | None = None) -> set[int]:
    """hadm_ids in-cohort = those with a diagnosis matching any cohort-filter
    concept. Empty filter -> every admission (from the `admissions` table).

    #132: when `dx_matcher` has an ICD family for a cohort concept, membership is
    decided by the ICD HIERARCHY on icd_code (A40*/A41*/R652*/...), else by the
    resolved (#109) codes."""
    if not plan.cohort_filter_concepts:
        adm = tables.get("admissions")
        return set(adm["hadm_id"].astype("int64")) if adm is not None else set()
    dx = tables.get("diagnoses_icd")
    if dx is None or dx.empty:
        return set()
    icd = dx["icd_code"].astype(str)
    import pandas as pd
    mask = pd.Series(False, index=dx.index)
    fam_codes: set[str] = set()
    for c in plan.cohort_filter_concepts:
        cs = (dx_matcher or {}).get(c)
        if cs is not None:
            mask = mask | cs.mask(icd)            # ICD-family (hierarchy) match
        else:
            fam_codes |= set(resolve(c))          # fallback: resolved enumerated codes
    if fam_codes:
        mask = mask | icd.isin(fam_codes)
    return set(dx[mask]["hadm_id"].astype("int64"))


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
            resolve: Resolver | None = None, drug_matcher: dict | None = None,
            dx_matcher: dict | None = None) -> pd.DataFrame:
    """Build the canonical 5-col stream for `plan` from the given MIMIC `tables`.

    `tables` keys are MIMIC table names (admissions, diagnoses_icd, labevents,
    chartevents, prescriptions, ...); only those needed by the plan are touched.
    `dx_matcher` (#132): {concept: IcdCodeSet} -> diagnosis cohort/eligibility match
    by ICD FAMILY (hierarchy) instead of name. `drug_matcher` (#131): meds by code.
    Returns a DataFrame that passes ``validate_canonical``.
    """
    resolve = resolve or _identity_resolver
    hadm_ids = _cohort_hadm_ids(plan, tables, resolve, dx_matcher)
    if not hadm_ids:
        return _empty_canonical()
    anchors = _anchor_times(plan, tables, hadm_ids)
    lo, hi = effective_window(plan)

    parts: list[pd.DataFrame] = []
    # ICU-ADMISSION ANCHOR (the t0 marker) — emit a LOCATION event at admittime per
    # cohort admission so the landmark t0 = admission, NOT the earliest event (which,
    # for a control/outcome-only trajectory, is the death event -> immortal-excluded
    # -> empty cohort + collapsed control arms). One LOCATION 'icu_admission' per hadm.
    if anchors:
        parts.append(pd.DataFrame({
            "TRAJECTORY_ID": [int(h) for h in anchors],
            "TIMESTAMP": pd.to_datetime(list(anchors.values()), utc=True),
            "EVENT_TYPE": EventType.LOCATION.value,
            "EVENT_NAME": "icu_admission",
            "EVENT_VALUE": "1",
        }))

    for req in plan.concepts:
        spec = TABLE_SPEC.get(req.event_type)
        if spec is None:
            continue
        # #131 seam A: when a drug_matcher is supplied, MEDICATION is matched BY CODE
        # (gsn/ndc) + emitted as the arm concept, NOT by name — handled below.
        if req.event_type == EventType.MEDICATION and drug_matcher:
            continue
        src = tables.get(spec["table"])
        if src is None or src.empty:
            continue
        in_cohort = src["hadm_id"].astype("int64").isin(hadm_ids)
        dx_fam = (dx_matcher or {}).get(req.concept) if req.event_type == EventType.DIAGNOSIS else None
        if dx_fam is not None:                          # #132: ICD-family match on icd_code
            name_match = dx_fam.mask(src[spec["name"]].astype(str))
        else:
            name_match = src[spec["name"]].astype(str).isin(resolve(req.concept))
        rows = src[in_cohort & name_match].copy()
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

    # OUTCOME events (e.g. death) come from admissions.deathtime, not a name-keyed
    # table, and are NOT clipped to the extraction window — they're post-baseline,
    # bounded later by the outcome horizon in the cohort builder.
    outcome_reqs = [r for r in plan.concepts if r.event_type == EventType.OUTCOME]
    adm = tables.get("admissions")
    if outcome_reqs and adm is not None and "deathtime" in adm.columns:
        dead = adm[adm["hadm_id"].astype("int64").isin(hadm_ids) & adm["deathtime"].notna()]
        if not dead.empty:
            for req in outcome_reqs:
                parts.append(pd.DataFrame({
                    "TRAJECTORY_ID": dead["hadm_id"].astype("int64"),
                    "TIMESTAMP": pd.to_datetime(dead["deathtime"], utc=True),
                    "EVENT_TYPE": EventType.OUTCOME.value,
                    "EVENT_NAME": req.concept,          # e.g. "death"
                    "EVENT_VALUE": "1",
                }))

    # #131 seam A: code-based MEDICATION events. Match prescriptions by gsn/ndc to the
    # arm concept's code set; emit EVENT_NAME=concept (so _assign_arm matches unchanged
    # + code-correct), EVENT_VALUE=JSON {raw_name, code, method, dose, source_table}.
    rx = tables.get("prescriptions")
    if drug_matcher and rx is not None and not rx.empty:
        from tteEngine.matching import assign_med_concepts, med_event_value
        rows = rx[rx["hadm_id"].astype("int64").isin(hadm_ids)].copy()
        if not rows.empty:
            concept, code, method = assign_med_concepts(rows, ("gsn", "ndc"), drug_matcher)
            rows = rows.assign(_concept=concept, _code=code, _method=method)
            rows = rows[rows["_concept"].notna()]
            rows["__ts"] = pd.to_datetime(rows.get("starttime"), utc=True, errors="coerce")
            if anchors and not rows.empty:
                anc = rows["hadm_id"].astype("int64").map(anchors)
                rows = rows[anc.notna() & (rows["__ts"] >= anc + pd.to_timedelta(lo, "h"))
                            & (rows["__ts"] <= anc + pd.to_timedelta(hi, "h"))]
            if not rows.empty:
                n = len(rows)
                names = rows["drug"].tolist() if "drug" in rows.columns else [None] * n
                doses = rows["dose_val_rx"].tolist() if "dose_val_rx" in rows.columns else [None] * n
                values = [med_event_value(raw_name=nm, code=cd, method=mt, dose=ds,
                                          source_table="prescriptions")
                          for nm, cd, mt, ds in zip(names, rows["_code"], rows["_method"], doses)]
                parts.append(pd.DataFrame({
                    "TRAJECTORY_ID": rows["hadm_id"].astype("int64"),
                    "TIMESTAMP": rows["__ts"],
                    "EVENT_TYPE": EventType.MEDICATION.value,
                    "EVENT_NAME": rows["_concept"].astype(str),          # the matched arm concept
                    "EVENT_VALUE": values,
                }))

    if not parts:
        return _empty_canonical()
    df = pd.concat(parts, ignore_index=True)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    df = df.sort_values(["TRAJECTORY_ID", "TIMESTAMP"]).reset_index(drop=True)
    df = harmonize_timestamps(df, getattr(plan, "timing", None))  # #31 common precision grid
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
