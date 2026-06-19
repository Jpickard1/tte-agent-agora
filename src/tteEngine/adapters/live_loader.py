"""Plan-targeted real-data loaders (#101, worker1) — the live-run enabler.

Read the real MIMIC-IV 3.1 + eICU-CRD 2.0 gz CSVs into the injected `tables`
dicts that adapters.mimic.extract / adapters.eicu.extract consume — PLAN-TARGETED
(only the needed tables, only the needed columns, row-filtered by the plan's
cohort/concept codes) and READ-ONLY by path. Big tables (labevents, chartevents,
vitalPeriodic/Aperiodic) are read in CHUNKS so memory stays bounded.

The loader is a targeted PREFETCH, not the source of truth: extract() re-filters
(cohort, window, concept) on what's handed to it, so the loader only has to avoid
UNDER-fetching. Two real-vs-fixture reconciliations are handled here so extract()
sees its expected shape:
  * MIMIC diagnoses_icd has NO per-diagnosis timestamp -> we derive `charttime`
    from the admission's admittime;
  * MIMIC labevents/chartevents key on `itemid` -> we join d_labitems/d_items to
    attach the `label` column the adapter matches on.
eICU's real columns already match the adapter's TABLE_SPEC, so its load is mostly
targeted reads (+ the wide vitals tables, column-targeted).

Paths default to the exxact07 mounts and are overridable via env (TTE_MIMIC_ROOT /
TTE_EICU_ROOT) for other hosts. pandas only (no extra deps).
"""
from __future__ import annotations

import os
from typing import Callable

import pandas as pd

from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ExtractionPlan

MIMIC_ROOT = os.environ.get("TTE_MIMIC_ROOT", "/ewsc/ewsc/clinical_data/mimiciv/3.1")
EICU_ROOT = os.environ.get(
    "TTE_EICU_ROOT", "/ewsc/ewsc/clinical_data/physionet.org/files/eicu-crd/2.0")

Resolver = Callable[[str], set]
_CHUNK = 2_000_000


def _identity(concept: str) -> set:
    return {concept}


def _names_for(plan: ExtractionPlan, event_type: EventType, resolve: Resolver) -> set:
    """All source names/codes the plan needs for one event type (concept + its
    resolved codes)."""
    out: set = set()
    for c in plan.concepts:
        if c.event_type == event_type:
            out |= set(resolve(c.concept)) | {c.concept}
    return out


def _read_filtered(path: str, *, usecols, keep, dtype=None, chunksize: int = _CHUNK) -> pd.DataFrame:
    """Chunked read of a (large) gz CSV, concatenating only rows where keep(chunk)
    is True. Bounded memory; returns an empty (typed-cols) frame if nothing matches."""
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=chunksize):
        sub = chunk[keep(chunk)]
        if len(sub):
            parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(usecols))


# --------------------------------------------------------------------------- #
# MIMIC-IV
# --------------------------------------------------------------------------- #
def load_mimic(plan: ExtractionPlan, *, root: str = MIMIC_ROOT,
               resolve: Resolver | None = None, chunksize: int = _CHUNK) -> dict:
    """Load the plan-targeted MIMIC-IV tables for adapters.mimic.extract."""
    resolve = resolve or _identity
    hosp, icu = f"{root}/hosp", f"{root}/icu"
    types = {c.event_type for c in plan.concepts}
    tables: dict[str, pd.DataFrame] = {}

    # admissions: cohort universe + anchor (admittime) + mortality (deathtime)
    adm = pd.read_csv(f"{hosp}/admissions.csv.gz", usecols=["hadm_id", "admittime", "deathtime"])
    tables["admissions"] = adm
    admit_by_hadm = dict(zip(adm["hadm_id"].astype("int64"), adm["admittime"]))
    all_hadm = set(adm["hadm_id"].astype("int64"))

    cohort_codes: set = set()
    for c in plan.cohort_filter_concepts:
        cohort_codes |= set(resolve(c))
    want_dx = cohort_codes | _names_for(plan, EventType.DIAGNOSIS, resolve)

    cohort_hadm = all_hadm
    if want_dx:
        dx = pd.read_csv(f"{hosp}/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code"],
                         dtype={"icd_code": str})
        dx = dx[dx["icd_code"].astype(str).isin(want_dx)].copy()
        dx["charttime"] = dx["hadm_id"].astype("int64").map(admit_by_hadm)  # no native dx time
        tables["diagnoses_icd"] = dx
        if cohort_codes:
            cohort_hadm = set(dx[dx["icd_code"].isin(cohort_codes)]["hadm_id"].astype("int64"))

    lab_names = {n.lower() for n in _names_for(plan, EventType.LAB, resolve)}
    if lab_names:
        dlab = pd.read_csv(f"{hosp}/d_labitems.csv.gz", usecols=["itemid", "label"])
        sel = dlab[dlab["label"].astype(str).str.lower().isin(lab_names)]
        id2label, ids = dict(zip(sel["itemid"], sel["label"])), set(sel["itemid"])
        if ids:
            lab = _read_filtered(
                f"{hosp}/labevents.csv.gz", usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                keep=lambda c: c["itemid"].isin(ids) & c["hadm_id"].isin(cohort_hadm),
                chunksize=chunksize)
            lab["label"] = lab["itemid"].map(id2label)
            tables["labevents"] = lab

    meas_names = {n.lower() for n in _names_for(plan, EventType.MEASUREMENT, resolve)}
    if meas_names:
        ditems = pd.read_csv(f"{icu}/d_items.csv.gz", usecols=["itemid", "label"])
        sel = ditems[ditems["label"].astype(str).str.lower().isin(meas_names)]
        id2label, ids = dict(zip(sel["itemid"], sel["label"])), set(sel["itemid"])
        if ids:
            ce = _read_filtered(
                f"{icu}/chartevents.csv.gz", usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                keep=lambda c: c["itemid"].isin(ids) & c["hadm_id"].isin(cohort_hadm),
                chunksize=chunksize)
            ce["label"] = ce["itemid"].map(id2label)
            tables["chartevents"] = ce

    med_names = {n.lower() for n in _names_for(plan, EventType.MEDICATION, resolve)}
    if med_names:
        tables["prescriptions"] = _read_filtered(
            f"{hosp}/prescriptions.csv.gz", usecols=["hadm_id", "drug", "starttime", "dose_val_rx"],
            keep=lambda c: c["drug"].astype(str).str.lower().isin(med_names)
            & c["hadm_id"].isin(cohort_hadm), chunksize=chunksize)
    return tables


# --------------------------------------------------------------------------- #
# eICU-CRD
# --------------------------------------------------------------------------- #
#: adapter table-key -> real eICU file basename (the vitals files are camelCase).
_EICU_VITAL_FILES = {"vitalperiodic": "vitalPeriodic.csv.gz", "vitalaperiodic": "vitalAperiodic.csv.gz"}


def load_eicu(plan: ExtractionPlan, *, root: str = EICU_ROOT,
              resolve: Resolver | None = None, chunksize: int = _CHUNK) -> dict:
    """Load the plan-targeted eICU-CRD tables for adapters.eicu.extract."""
    resolve = resolve or _identity
    types = {c.event_type for c in plan.concepts}
    tables: dict[str, pd.DataFrame] = {}

    # patient: cohort universe + stays + mortality (discharge status/offset)
    patient = pd.read_csv(f"{root}/patient.csv.gz", usecols=[
        "patientunitstayid", "unitdischargestatus", "unitdischargeoffset",
        "hospitaldischargestatus", "hospitaldischargeoffset"])
    tables["patient"] = patient
    all_stays = set(patient["patientunitstayid"].astype("int64"))

    cohort_codes: set = set()
    for c in plan.cohort_filter_concepts:
        cohort_codes |= set(resolve(c))
    want_dx = cohort_codes | _names_for(plan, EventType.DIAGNOSIS, resolve)

    cohort_stays = all_stays
    if want_dx or plan.cohort_filter_concepts:
        dx = pd.read_csv(f"{root}/diagnosis.csv.gz",
                         usecols=["patientunitstayid", "diagnosisoffset", "icd9code", "diagnosisstring"],
                         dtype={"icd9code": str})
        if want_dx:
            # eICU icd9code can pack several codes ("A41.9, 995.92") -> keep a row if
            # ANY requested code is a substring (liberal prefetch; extract re-matches).
            codes = list(want_dx)
            dx = dx[dx["icd9code"].astype(str).apply(lambda s: any(k in s for k in codes))].copy()
        tables["diagnosis"] = dx
        if cohort_codes:
            cc = list(cohort_codes)
            cohort_stays = set(dx[dx["icd9code"].astype(str).apply(
                lambda s: any(k in s for k in cc))]["patientunitstayid"].astype("int64"))

    lab_names = {n.lower() for n in _names_for(plan, EventType.LAB, resolve)}
    if lab_names:
        tables["lab"] = _read_filtered(
            f"{root}/lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
            keep=lambda c: c["labname"].astype(str).str.lower().isin(lab_names)
            & c["patientunitstayid"].isin(cohort_stays), chunksize=chunksize)

    med_names = {n.lower() for n in _names_for(plan, EventType.MEDICATION, resolve)}
    if med_names:
        tables["medication"] = _read_filtered(
            f"{root}/medication.csv.gz", usecols=["patientunitstayid", "drugstartoffset", "drugname", "dosage"],
            keep=lambda c: c["drugname"].astype(str).str.lower().isin(med_names)
            & c["patientunitstayid"].isin(cohort_stays), chunksize=chunksize)

    if EventType.MEASUREMENT in types:
        from tteEngine.adapters.eicu import VITAL_COLUMNS, VITAL_OFFSET
        wanted = _names_for(plan, EventType.MEASUREMENT, resolve)
        want_cols = {col for col, vit in VITAL_COLUMNS.items() if vit in wanted}
        for key, fname in _EICU_VITAL_FILES.items():
            path = f"{root}/{fname}"
            if not os.path.exists(path):
                continue
            # VITAL_COLUMNS spans BOTH vital tables -> read only the cols this file has
            avail = set(pd.read_csv(path, nrows=0).columns)
            cols = [c for c in want_cols if c in avail]
            if not cols:
                continue
            tables[key] = _read_filtered(
                path, usecols=["patientunitstayid", VITAL_OFFSET, *cols],
                keep=lambda c: c["patientunitstayid"].isin(cohort_stays), chunksize=chunksize)
    return tables


# --------------------------------------------------------------------------- #
# Live-run seam (#102): extract_fn(plan, spec, dataset) -> canonical 5-col | None
# --------------------------------------------------------------------------- #
def make_extract_fn(datasets=None, *, resolve=None, mimic_root: str = MIMIC_ROOT,
                    eicu_root: str = EICU_ROOT, chunksize: int = _CHUNK):
    """Return the `extract_fn(plan, spec, dataset) -> canonical 5-col DataFrame |
    None` that probe's #102 run_corpus consumes: load the plan-targeted real tables
    for `dataset` and run the matching adapter.

    `resolve` (concept -> source codes, default vocab.resolve) feeds BOTH the loader
    (row pre-filter) and the adapter (concept match). Datasets not handled here
    (MGB — human-gated; or any outside `datasets`) return None, which run_corpus
    logs as a drop (no silent cap)."""
    from tteEngine import vocab
    from tteEngine.adapters import eicu, mimic

    _resolve = resolve or vocab.resolve
    allowed = set(datasets) if datasets else None

    def extract_fn(plan, spec, dataset):
        if allowed is not None and dataset not in allowed:
            return None
        if dataset == "MIMIC-IV":
            df = mimic.extract(plan, load_mimic(plan, root=mimic_root, resolve=_resolve,
                                                chunksize=chunksize), resolve=_resolve)
        elif dataset == "eICU-CRD":
            df = eicu.extract(plan, load_eicu(plan, root=eicu_root, resolve=_resolve,
                                              chunksize=chunksize), resolve=_resolve)
        else:
            return None  # MGB is human-gated (#8); unknown datasets unsupported
        return df if df is not None and not df.empty else None

    return extract_fn


__all__ = ["load_mimic", "load_eicu", "make_extract_fn", "MIMIC_ROOT", "EICU_ROOT"]
