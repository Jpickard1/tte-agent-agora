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
import re
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
               resolve: Resolver | None = None, chunksize: int = _CHUNK, prepass=None,
               drug_matcher: dict | None = None, dx_matcher: dict | None = None) -> dict:
    """Load the plan-targeted MIMIC-IV tables for adapters.mimic.extract.

    `prepass` (optional #124 Prepass): when given, labevents/chartevents come from
    the shared pre-pass sliced to this cohort (no per-trial rescan of the 2.5/3.3GB
    tables) — what makes lab COVARIATES feasible at >=1k scale.
    `drug_matcher` (optional #131): when given, prescriptions are loaded by CODE
    (gsn/ndc in the matcher's code union) rather than by drug name, so the adapter
    can code-match meds (seam A)."""
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
    if want_dx or dx_matcher:
        dx = pd.read_csv(f"{hosp}/diagnoses_icd.csv.gz", usecols=["hadm_id", "icd_code"],
                         dtype={"icd_code": str})
        icd = dx["icd_code"].astype(str)
        keep = icd.isin(want_dx)
        fam_mask = pd.Series(False, index=dx.index)        # #132 ICD-family rows
        for cs in (dx_matcher or {}).values():
            fam_mask = fam_mask | cs.mask(icd)
        dx = dx[keep | fam_mask].copy()
        dx["charttime"] = dx["hadm_id"].astype("int64").map(admit_by_hadm)  # no native dx time
        tables["diagnoses_icd"] = dx
        # cohort = family matches (if any) OR the resolved cohort codes
        if dx_matcher or cohort_codes:
            cm = pd.Series(False, index=dx.index)
            for cs in (dx_matcher or {}).values():
                cm = cm | cs.mask(dx["icd_code"].astype(str))
            if cohort_codes:
                cm = cm | dx["icd_code"].astype(str).isin(cohort_codes)
            cohort_hadm = set(dx[cm]["hadm_id"].astype("int64"))

    # labs/measurements: from the shared pre-pass (sliced, no rescan) if given, else
    # the per-trial filtered scan.
    pp_tables = prepass.slice(cohort_hadm) if prepass is not None else {}

    lab_names = {n.lower() for n in _names_for(plan, EventType.LAB, resolve)}
    if lab_names:
        if "labevents" in pp_tables:
            tables["labevents"] = pp_tables["labevents"]
        else:
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
        if "chartevents" in pp_tables:
            tables["chartevents"] = pp_tables["chartevents"]
        else:
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

    pcols = ["hadm_id", "drug", "starttime", "dose_val_rx", "gsn", "ndc"]
    if drug_matcher:  # #131: load cohort prescriptions whose CODE is in the matcher union
        from tteEngine.matching import _norm_code
        codeunion = {k for cs in drug_matcher.values() for k in cs.codes}
        if "prescriptions" in pp_tables:
            # #171: the drug pre-pass already scanned prescriptions ONCE (corpus-union
            # codes); pp_tables is this cohort's slice. The adapter's drug_matcher
            # narrows to THIS trial's arm codes, so the result is byte-identical to the
            # per-trial read below — without re-decompressing the 579MB file each trial.
            tables["prescriptions"] = pp_tables["prescriptions"]
        elif codeunion:
            tables["prescriptions"] = _read_filtered(
                f"{hosp}/prescriptions.csv.gz", usecols=pcols,
                keep=lambda c: c["hadm_id"].isin(cohort_hadm)
                & (c["gsn"].map(_norm_code).isin(codeunion) | c["ndc"].map(_norm_code).isin(codeunion)),
                chunksize=chunksize)
    else:
        med_names = {n.lower() for n in _names_for(plan, EventType.MEDICATION, resolve)}
        if med_names:
            tables["prescriptions"] = _read_filtered(
                f"{hosp}/prescriptions.csv.gz", usecols=pcols,
                keep=lambda c: c["drug"].astype(str).str.lower().isin(med_names)
                & c["hadm_id"].isin(cohort_hadm), chunksize=chunksize)
    return tables


# --------------------------------------------------------------------------- #
# eICU-CRD
# --------------------------------------------------------------------------- #
#: adapter table-key -> real eICU file basename (the vitals files are camelCase).
_EICU_VITAL_FILES = {"vitalperiodic": "vitalPeriodic.csv.gz", "vitalaperiodic": "vitalAperiodic.csv.gz"}


def load_eicu(plan: ExtractionPlan, *, root: str = EICU_ROOT,
              resolve: Resolver | None = None, chunksize: int = _CHUNK, prepass=None,
              drug_matcher: dict | None = None, dx_matcher: dict | None = None) -> dict:
    """Load the plan-targeted eICU-CRD tables for adapters.eicu.extract.

    `prepass` (optional #124 Prepass): lab/vitalperiodic come from the shared
    pre-pass sliced to this cohort (no per-trial rescan) when supplied.
    `drug_matcher` (optional #131): medication loaded by CODE (drughiclseqno in the
    matcher's code union) rather than by drug name. `dx_matcher` (#132): diagnoses
    prefetched by ICD family (hierarchy) too, so the adapter's family match sees them."""
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
    if want_dx or dx_matcher or plan.cohort_filter_concepts:
        dx = pd.read_csv(f"{root}/diagnosis.csv.gz",
                         usecols=["patientunitstayid", "diagnosisoffset", "icd9code", "diagnosisstring"],
                         dtype={"icd9code": str})
        icd = dx["icd9code"].astype(str)
        fams = list((dx_matcher or {}).values())
        if want_dx or fams:
            codes = list(want_dx)
            # keep a row if ANY code token substring-matches want_dx OR is in a #132 ICD family
            keep = icd.apply(lambda s: any(k in s for k in codes)) if codes else pd.Series(False, index=dx.index)
            for cs in fams:
                keep = keep | icd.map(cs.matches_any)
            dx = dx[keep].copy()
        tables["diagnosis"] = dx
        if dx_matcher or cohort_codes:
            cm = pd.Series(False, index=dx.index)
            for cs in fams:
                cm = cm | dx["icd9code"].astype(str).map(cs.matches_any)
            if cohort_codes:
                cc = list(cohort_codes)
                cm = cm | dx["icd9code"].astype(str).apply(lambda s: any(k in s for k in cc))
            cohort_stays = set(dx[cm]["patientunitstayid"].astype("int64"))

    pp_tables = prepass.slice(cohort_stays) if prepass is not None else {}

    lab_names = {n.lower() for n in _names_for(plan, EventType.LAB, resolve)}
    if lab_names:
        if "lab" in pp_tables:
            tables["lab"] = pp_tables["lab"]
        else:
            tables["lab"] = _read_filtered(
                f"{root}/lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
                keep=lambda c: c["labname"].astype(str).str.lower().isin(lab_names)
                & c["patientunitstayid"].isin(cohort_stays), chunksize=chunksize)

    mcols = ["patientunitstayid", "drugstartoffset", "drugname", "dosage", "drughiclseqno"]
    if drug_matcher:  # #131: load cohort medication whose HICL code is in the matcher union
        from tteEngine.matching import _norm_code
        codeunion = {k for cs in drug_matcher.values() for k in cs.codes}
        if "medication" in pp_tables:
            tables["medication"] = pp_tables["medication"]   # #171: corpus-union pre-pass slice
        elif codeunion:
            tables["medication"] = _read_filtered(
                f"{root}/medication.csv.gz", usecols=mcols,
                keep=lambda c: c["patientunitstayid"].isin(cohort_stays)
                & c["drughiclseqno"].map(_norm_code).isin(codeunion), chunksize=chunksize)
    else:
        med_names = {n.lower() for n in _names_for(plan, EventType.MEDICATION, resolve)}
        if med_names:
            tables["medication"] = _read_filtered(
                f"{root}/medication.csv.gz", usecols=mcols,
                keep=lambda c: c["drugname"].astype(str).str.lower().isin(med_names)
                & c["patientunitstayid"].isin(cohort_stays), chunksize=chunksize)

    if EventType.MEASUREMENT in types and "vitalperiodic" in pp_tables:
        tables["vitalperiodic"] = pp_tables["vitalperiodic"]   # #124 pre-pass slice
    elif EventType.MEASUREMENT in types:
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
#: EventType -> #109 vocab-index category for auto-resolving concepts to real codes.
_INDEX_CATEGORY = {
    EventType.DIAGNOSIS: "diagnosis", EventType.LAB: "lab",
    EventType.MEDICATION: "medication", EventType.MEASUREMENT: "vital",
}


def _enrich_vocab_from_index(plan: ExtractionPlan, index: dict) -> None:
    """Register the plan's concepts onto the #5 vocab using the #109 per-dataset
    index, so vocab.resolve (concept->codes) AND vocab.classify (code->concept)
    return the DATA-DRIVEN real codes for this dataset — what makes real ctgov
    cohorts non-empty (a free-text condition like 'Sepsis' otherwise resolves to
    nothing). Tries the full concept phrase first (precise), else its alpha tokens."""
    from tteEngine import vocab
    from tteEngine.adapters.vocab_index import codes_for

    def reg(concept: str, category: str) -> None:
        if not concept:
            return
        codes = codes_for(index, category, concept.lower())
        if not codes:  # fall back to meaningful alpha tokens ('Sepsis-3' -> 'sepsis')
            toks = [t for t in re.split(r"[^a-z0-9]+", concept.lower()) if len(t) >= 4]
            if toks:
                codes = codes_for(index, category, *toks)
        if codes:
            vocab.register_concept(concept, codes)

    for concept in plan.cohort_filter_concepts:
        reg(concept, "diagnosis")
    for req in plan.concepts:
        cat = _INDEX_CATEGORY.get(req.event_type)
        if cat:
            reg(req.concept, cat)


def _merge_drug_prepass(existing, dataset, specs, *, root, drug_catalog_cache,
                        drug_prepass_cache, refresh):
    """#171: build the corpus-union drug pre-pass for `dataset` and fold it into any
    `existing` Prepass (e.g. a full-mode lab/chart pre-pass) so one Prepass carries
    every pre-scanned table. Returns the (possibly new) Prepass, or `existing` if the
    corpus has no resolvable drug codes."""
    from tteEngine.adapters import prepass as PP
    codes = PP.corpus_drug_codes(specs, dataset, root=root, cache_dir=drug_catalog_cache)
    if not codes:
        return existing
    kw = {"drug_codes": codes, "root": root, "refresh": refresh}
    if drug_prepass_cache is not None:
        kw["drug_cache_dir"] = drug_prepass_cache
    pp = PP.build_mimic_prepass(**kw) if dataset == "MIMIC-IV" else PP.build_eicu_prepass(**kw)
    if existing is not None:                     # keep the lab/chart tables too
        existing.tables.update(pp.tables)
        return existing
    return pp


def make_extract_fn(datasets=None, *, resolve=None, mimic_root: str = MIMIC_ROOT,
                    eicu_root: str = EICU_ROOT, chunksize: int = _CHUNK,
                    use_vocab_index: bool = True, index_cache_dir=None, prepasses=None,
                    use_drug_codes: bool = True, drug_catalog_cache=None,
                    use_dx_codes: bool = True, drug_prepass: bool = False,
                    corpus_specs=None, drug_prepass_cache=None, prepass_refresh: bool = False):
    """Return the `extract_fn(plan, spec, dataset) -> canonical 5-col DataFrame |
    None` that probe's #102 run_corpus consumes: load the plan-targeted real tables
    for `dataset` and run the matching adapter.

    `resolve` (concept -> source codes, default vocab.resolve) feeds BOTH the loader
    (row pre-filter) and the adapter (concept match). With `use_vocab_index` (default
    on), the per-dataset #109 index is built/loaded ONCE and each trial's concepts
    are auto-registered onto the vocab before extraction — so a real ctgov condition
    ('Sepsis') resolves to that dataset's real codes (177 sepsis codes, not the
    curated 44) and the cohort is non-empty. Datasets not handled here (MGB —
    human-gated; or any outside `datasets`) return None (run_corpus logs the drop).

    `drug_prepass=True` + `corpus_specs` (#171): scan the big prescriptions/medication
    tables ONCE, pre-filtered to the corpus-union of arm drug codes, cached on /ewsc;
    each trial then SLICES its cohort instead of re-decompressing the 579MB file —
    so a >10k-trial run (#111) isn't bottlenecked. PURE caching: byte-identical
    cohort/arm assignment to the per-trial read (the adapter still narrows to each
    trial's arm codes)."""
    from tteEngine import vocab
    from tteEngine.adapters import eicu, mimic

    _resolve = resolve or vocab.resolve
    allowed = set(datasets) if datasets else None
    roots = {"MIMIC-IV": mimic_root, "eICU-CRD": eicu_root}

    if drug_prepass and corpus_specs:           # #171: build the corpus-union drug pre-pass ONCE
        prepasses = dict(prepasses or {})
        for ds in (datasets or ("MIMIC-IV", "eICU-CRD")):
            if ds in roots:
                try:
                    prepasses[ds] = _merge_drug_prepass(
                        prepasses.get(ds), ds, corpus_specs, root=roots[ds],
                        drug_catalog_cache=drug_catalog_cache,
                        drug_prepass_cache=drug_prepass_cache, refresh=prepass_refresh)
                except Exception:
                    pass  # degrade to the per-trial read (still correct, just slower)

    indexes: dict[str, dict] = {}
    if use_vocab_index:
        from tteEngine.adapters import vocab_index as VI
        for ds in (datasets or ("MIMIC-IV", "eICU-CRD")):
            if ds in roots:
                try:
                    kw = {"cache_dir": index_cache_dir} if index_cache_dir else {}
                    indexes[ds] = VI.build_vocab_index(ds, root=roots[ds], **kw)
                except Exception:
                    pass  # degrade gracefully to the curated vocab

    def _matcher(spec, dataset):
        if not (use_drug_codes and spec is not None and dataset in roots):
            return None
        try:  # cached drug catalog -> {arm_concept: DrugCodeSet} (#131 seam A)
            from tteEngine.matching import build_drug_matcher
            from tteEngine.matching import DRUG_CATALOG_CACHE
            return build_drug_matcher(spec, dataset, root=roots[dataset],
                                      cache_dir=drug_catalog_cache or DRUG_CATALOG_CACHE)
        except Exception:
            return None  # degrade to name matching

    def _dx(plan):
        if not use_dx_codes:
            return None
        from tteEngine.contracts.events import EventType
        from tteEngine.matching import build_dx_matcher
        concepts = list(plan.cohort_filter_concepts) + [
            c.concept for c in plan.concepts if c.event_type == EventType.DIAGNOSIS]
        return build_dx_matcher(concepts) or None

    def extract_fn(plan, spec, dataset):
        if allowed is not None and dataset not in allowed:
            return None
        idx = indexes.get(dataset)
        if idx is not None:
            _enrich_vocab_from_index(plan, idx)   # data-driven codes for this trial
        pp = (prepasses or {}).get(dataset)       # #124 shared pre-pass (full mode), if supplied
        dm = _matcher(spec, dataset)              # #131 code-based med matcher, if enabled
        dxm = _dx(plan)                           # #132 ICD-family cohort/dx matcher
        if dataset == "MIMIC-IV":
            df = mimic.extract(plan, load_mimic(plan, root=mimic_root, resolve=_resolve,
                                                chunksize=chunksize, prepass=pp, drug_matcher=dm,
                                                dx_matcher=dxm),
                               resolve=_resolve, drug_matcher=dm, dx_matcher=dxm)
        elif dataset == "eICU-CRD":
            df = eicu.extract(plan, load_eicu(plan, root=eicu_root, resolve=_resolve,
                                              chunksize=chunksize, prepass=pp, drug_matcher=dm,
                                              dx_matcher=dxm),
                              resolve=_resolve, drug_matcher=dm, dx_matcher=dxm)
        else:
            return None  # MGB is human-gated (#8); unknown datasets unsupported
        return df if df is not None and not df.empty else None

    return extract_fn


__all__ = ["load_mimic", "load_eicu", "make_extract_fn", "MIMIC_ROOT", "EICU_ROOT"]
