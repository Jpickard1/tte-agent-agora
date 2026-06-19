"""Shared lab/measurement PRE-PASS (#124, worker1) — scan the big tables ONCE.

At >=1k trials, re-scanning MIMIC labevents (2.5GB) + chartevents (3.3GB) per
trial (whenever a plan has lab/measurement covariates) is infeasible — that forced
lean mode (#111), which prunes lab covariates and weakens confounding adjustment.

This scans those big tables ONCE, pre-filtered to the UNION of lab/measurement
codes needed across the corpus, into a cached, patient-indexed store. Each trial
then SLICES its cohort's rows from the cache (in-memory, no rescan), so lab
COVARIATES come back at scale — restoring full adjustment (the 'pruned-for-scale'
confounders in the #105 ledger lift back to adjusted).

Flow (live_run 'full' mode):
    ids = prepass_itemids(corpus_lab_concepts(specs), index, 'lab')      # union, via #109
    pp  = build_mimic_prepass(lab_itemids=ids, root=...)                  # scan ONCE, cached
    load_mimic(plan, prepass=pp)  /  make_extract_fn(prepasses={ds: pp})  # per-trial = a slice

Read-only on the clinical data; cache (parquet) under $HOME, never /ewsc.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from tteEngine.adapters.live_loader import EICU_ROOT, MIMIC_ROOT, _CHUNK, _read_filtered
from tteEngine.contracts.events import EventType

PREPASS_CACHE = Path.home() / ".cache" / "tteEngine" / "prepass"


def corpus_lab_concepts(specs, *, event_types=(EventType.LAB, EventType.MEASUREMENT)) -> set[str]:
    """The lab/measurement concept strings the corpus needs (from eligibility
    criteria of those types + covariates)."""
    out: set[str] = set()
    for spec in specs:
        for c in spec.eligibility:
            if c.event_type in event_types and c.concept:
                out.add(c.concept)
        out |= {cov for cov in spec.covariates if cov}
    return out


def prepass_itemids(concepts, index: dict, category: str) -> set[str]:
    """Union of #109 catalog codes (MIMIC itemids / eICU labnames / vital cols) the
    given concepts resolve to — full phrase first, else alpha tokens. Pure."""
    entries = (index or {}).get("categories", {}).get(category, [])

    def hits(term: str) -> set[str]:
        t = term.lower()
        return {e["code"] for e in entries if t in e["name"].lower() or t in e["code"].lower()}

    out: set[str] = set()
    for concept in concepts:
        got = hits(concept)
        if not got:
            for tok in (t for t in re.split(r"[^a-z0-9]+", concept.lower()) if len(t) >= 4):
                got |= hits(tok)
        out |= got
    return out


@dataclass
class Prepass:
    """A cached, patient-indexed slice of the big tables (corpus-union rows). Per
    trial, ``slice(cohort_ids)`` returns adapter-shaped tables with no rescan."""
    dataset: str
    id_col: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def slice(self, cohort_ids) -> dict[str, pd.DataFrame]:
        ids = {int(x) for x in cohort_ids}
        return {name: df[df[self.id_col].astype("int64").isin(ids)].copy()
                for name, df in self.tables.items()}


def _cached(cache_dir: Path, name: str, refresh: bool, build):
    """Load a parquet from cache or build+write it."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = build()
    df.to_parquet(path, index=False)
    return df


def build_mimic_prepass(*, lab_itemids=(), chart_itemids=(), root: str = MIMIC_ROOT,
                        cache_dir: str | Path = PREPASS_CACHE, refresh: bool = False,
                        chunksize: int = _CHUNK) -> Prepass:
    """Scan MIMIC labevents/chartevents ONCE, filtered to the corpus-union itemids,
    `label`-joined + cached. Returns a Prepass keyed by hadm_id."""
    hosp, icu = f"{root}/hosp", f"{root}/icu"
    cache_dir = Path(cache_dir)
    pp = Prepass(dataset="MIMIC-IV", id_col="hadm_id")

    if lab_itemids:
        ids = set(lab_itemids)

        def _labs():  # source reads happen ONLY on a cache miss
            labels = pd.read_csv(f"{hosp}/d_labitems.csv.gz", usecols=["itemid", "label"])
            id2label = dict(zip(labels["itemid"].astype(str), labels["label"]))
            df = _read_filtered(f"{hosp}/labevents.csv.gz",
                                usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                                keep=lambda c: c["itemid"].astype(str).isin(ids), chunksize=chunksize)
            df["label"] = df["itemid"].astype(str).map(id2label)
            return df
        pp.tables["labevents"] = _cached(cache_dir, "mimic_labevents", refresh, _labs)

    if chart_itemids:
        ids = set(chart_itemids)

        def _chart():
            ditems = pd.read_csv(f"{icu}/d_items.csv.gz", usecols=["itemid", "label"])
            id2label = dict(zip(ditems["itemid"].astype(str), ditems["label"]))
            df = _read_filtered(f"{icu}/chartevents.csv.gz",
                                usecols=["hadm_id", "itemid", "charttime", "valuenum"],
                                keep=lambda c: c["itemid"].astype(str).isin(ids), chunksize=chunksize)
            df["label"] = df["itemid"].astype(str).map(id2label)
            return df
        pp.tables["chartevents"] = _cached(cache_dir, "mimic_chartevents", refresh, _chart)
    return pp


def build_eicu_prepass(*, labnames=(), vital_cols=(), root: str = EICU_ROOT,
                       cache_dir: str | Path = PREPASS_CACHE, refresh: bool = False,
                       chunksize: int = _CHUNK) -> Prepass:
    """Scan eICU lab / vitalPeriodic ONCE, filtered to the corpus-union labnames /
    vital columns, cached. Returns a Prepass keyed by patientunitstayid."""
    cache_dir = Path(cache_dir)
    pp = Prepass(dataset="eICU-CRD", id_col="patientunitstayid")

    if labnames:
        names = {n.lower() for n in labnames}

        def _lab():
            return _read_filtered(f"{root}/lab.csv.gz",
                                  usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"],
                                  keep=lambda c: c["labname"].astype(str).str.lower().isin(names),
                                  chunksize=chunksize)
        pp.tables["lab"] = _cached(cache_dir, "eicu_lab", refresh, _lab)

    if vital_cols:
        cols = [c for c in vital_cols]

        def _vit():
            return _read_filtered(f"{root}/vitalPeriodic.csv.gz",
                                  usecols=["patientunitstayid", "observationoffset", *cols],
                                  keep=lambda c: pd.Series(True, index=c.index), chunksize=chunksize)
        pp.tables["vitalperiodic"] = _cached(cache_dir, "eicu_vitalperiodic", refresh, _vit)
    return pp


__all__ = ["Prepass", "corpus_lab_concepts", "prepass_itemids",
           "build_mimic_prepass", "build_eicu_prepass", "PREPASS_CACHE"]
