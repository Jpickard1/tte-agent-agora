"""Per-dataset VOCABULARY INDEX (#109, worker1) — make concept->real-code
data-driven instead of curated.

Scans each DB's dictionaries / coded fields into a cached catalog of the
diagnoses / labs / meds / vitals actually present, with codes + names (+ counts
where cheap). This is the foundation that lets measurability (#33), triage (#35)
and the confounder ledger (#105) resolve a concept ('sepsis', 'lactate') to the
REAL codes in each dataset — e.g. it returns MIMIC's no-dot ICD-9 AND eICU's
DOTTED ICD-9 sepsis codes, fixing the format mismatch the live loaders hit.

Sources:
  MIMIC-IV: d_icd_diagnoses (+ diagnoses_icd counts), d_labitems, d_items (chart),
            prescriptions.drug.
  eICU-CRD: diagnosis (icd9code/diagnosisstring), lab.labname, medication.drugname,
            + the fixed vitalPeriodic/Aperiodic columns.

Read-only on the clinical data; the catalog is CACHED as JSON under $HOME
(never under /ewsc). Big tables are scanned chunked; lab/vital frequencies on the
>100M-row tables are skipped by default (count=None) — the dictionaries are the
catalog there. pandas only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from tteEngine.adapters.live_loader import EICU_ROOT, MIMIC_ROOT

DEFAULT_CACHE = Path.home() / ".cache" / "tteEngine" / "vocab_index"
CATEGORIES = ("diagnosis", "lab", "medication", "vital")
_CHUNK = 1_000_000


def _entries(codes, names, counts=None) -> list[dict]:
    out = []
    for code, name in zip(codes, names):
        e = {"code": str(code), "name": str(name)}
        if counts is not None:
            e["count"] = int(counts.get(code, 0))
        out.append(e)
    return out


def _chunked_value_counts(path, col, *, dtype=None) -> "pd.Series":
    """value_counts of one column over a (large) gz CSV, chunked."""
    total = None
    for chunk in pd.read_csv(path, usecols=[col], dtype=dtype, chunksize=_CHUNK):
        vc = chunk[col].astype(str).value_counts()
        total = vc if total is None else total.add(vc, fill_value=0)
    return total.astype(int) if total is not None else pd.Series(dtype=int)


def _build_mimic(root: str, with_counts: bool) -> dict:
    hosp, icu = f"{root}/hosp", f"{root}/icu"
    cats: dict[str, list] = {}
    # diagnosis: dictionary names + (cheap) per-code frequency from diagnoses_icd
    dd = pd.read_csv(f"{hosp}/d_icd_diagnoses.csv.gz", usecols=["icd_code", "long_title"],
                     dtype={"icd_code": str})
    counts = None
    if with_counts:
        counts = _chunked_value_counts(f"{hosp}/diagnoses_icd.csv.gz", "icd_code",
                                       dtype={"icd_code": str}).to_dict()
    cats["diagnosis"] = _entries(dd["icd_code"], dd["long_title"], counts)
    # labs / measurements: dictionaries are the catalog (event-table counts skipped)
    dlab = pd.read_csv(f"{hosp}/d_labitems.csv.gz", usecols=["itemid", "label"])
    cats["lab"] = _entries(dlab["itemid"], dlab["label"])
    ditems = pd.read_csv(f"{icu}/d_items.csv.gz", usecols=["itemid", "label", "linksto"])
    chart = ditems[ditems["linksto"].astype(str) == "chartevents"]
    cats["vital"] = _entries(chart["itemid"], chart["label"])
    # medications: no dictionary -> distinct drug names (+ counts) from prescriptions
    drug_counts = _chunked_value_counts(f"{hosp}/prescriptions.csv.gz", "drug")
    cats["medication"] = _entries(drug_counts.index, drug_counts.index,
                                  drug_counts.to_dict() if with_counts else None)
    return cats


def _build_eicu(root: str, with_counts: bool) -> dict:
    cats: dict[str, list] = {}
    # diagnosis: no dict -> (icd9code, diagnosisstring) catalog from the table, with counts
    seen: dict[str, dict] = {}
    for chunk in pd.read_csv(f"{root}/diagnosis.csv.gz",
                             usecols=["icd9code", "diagnosisstring"],
                             dtype={"icd9code": str}, chunksize=_CHUNK):
        for code, name in zip(chunk["icd9code"].astype(str), chunk["diagnosisstring"].astype(str)):
            for tok in (t.strip() for t in code.replace(";", ",").split(",") if t.strip()):
                e = seen.setdefault(tok, {"code": tok, "name": name, "count": 0})
                e["count"] += 1
    cats["diagnosis"] = list(seen.values()) if with_counts else \
        [{"code": e["code"], "name": e["name"]} for e in seen.values()]
    # labs / meds: distinct names (+counts) from the coded fields
    lab = _chunked_value_counts(f"{root}/lab.csv.gz", "labname")
    cats["lab"] = _entries(lab.index, lab.index, lab.to_dict() if with_counts else None)
    med = _chunked_value_counts(f"{root}/medication.csv.gz", "drugname")
    cats["medication"] = _entries(med.index, med.index, med.to_dict() if with_counts else None)
    # vitals: a FIXED column set (no scan needed)
    from tteEngine.adapters.eicu import VITAL_COLUMNS
    cats["vital"] = [{"code": col, "name": vit} for col, vit in VITAL_COLUMNS.items()]
    return cats


def build_vocab_index(dataset: str, *, root: str | None = None, with_counts: bool = True,
                      cache_dir: str | Path = DEFAULT_CACHE, refresh: bool = False) -> dict:
    """Build (or load from cache) the per-dataset vocabulary index. Returns
    ``{"dataset", "built_at", "categories": {category: [{code, name, count?}]}}``.
    Cached as JSON under $HOME; pass refresh=True to rebuild."""
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{dataset}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    if dataset == "MIMIC-IV":
        cats = _build_mimic(root or MIMIC_ROOT, with_counts)
    elif dataset == "eICU-CRD":
        cats = _build_eicu(root or EICU_ROOT, with_counts)
    else:
        raise ValueError(f"no vocab-index builder for dataset {dataset!r}")

    index = {"dataset": dataset, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "categories": cats}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index))
    return index


def search(index: dict, category: str, term: str, *, limit: int = 50) -> list[dict]:
    """Catalog entries in `category` whose name OR code contains `term` (case-
    insensitive), most frequent first when counts are present."""
    t = term.lower()
    hits = [e for e in index["categories"].get(category, [])
            if t in e["name"].lower() or t in e["code"].lower()]
    hits.sort(key=lambda e: -e.get("count", 0))
    return hits[:limit]


def codes_for(index: dict, category: str, *terms: str) -> set[str]:
    """The REAL codes in `category` whose name/code matches ANY term — the
    data-driven concept->codes resolution (e.g. codes_for(idx,'diagnosis','sepsis',
    'septic') -> every sepsis ICD code in that dataset, dotted or not)."""
    out: set[str] = set()
    for term in terms:
        out |= {e["code"] for e in search(index, category, term, limit=10_000)}
    return out


def register_into_vocab(index: dict, concept: str, category: str, *terms: str) -> set[str]:
    """Resolve `terms` to this dataset's real codes via the index and register them
    on the #5 vocab concept (so adapters/cohort match them). Returns the codes added."""
    from tteEngine import vocab

    codes = codes_for(index, category, *(terms or (concept,)))
    if codes:
        vocab.register_concept(concept, codes)
    return codes


__all__ = ["build_vocab_index", "search", "codes_for", "register_into_vocab",
           "DEFAULT_CACHE", "CATEGORIES"]
