"""Corpus builder (#1 extension, probe) — the bridge from the ctgov reader to the
emulability catalog (#35).

Fetches a large, SEPSIS-PRIORITIZED corpus of COMPLETED + results-posted trials
(paged via the #1 reader, cached, deduplicated), so the catalog (#35) can be
built over >1000 (configurable to >10k) real trials. Self-contained: code in-repo,
trials fetched live from CT.gov + cached (offline-replayable after first fetch).

    from tteEngine.ctgov import fetch_corpus, study_to_spec
    studies = fetch_corpus(max_studies=2000, sepsis_first=True)
    specs = [study_to_spec(s) for s in studies]   # -> #35 score_spec / build_catalog
"""
from __future__ import annotations

from typing import Iterator

from .reader import DEFAULT_CACHE, fetch_batch, nct_id_of
from .spec import study_to_spec

# Conditions pulled first so the corpus is sepsis-weighted (jpic's priority).
SEPSIS_TERMS = ("sepsis", "septic shock")


def fetch_corpus(
    *,
    max_studies: int = 2000,
    sepsis_first: bool = True,
    require_results: bool = True,
    overall_status: str = "COMPLETED",
    cache_dir=DEFAULT_CACHE,
    http_get=None,
) -> list[dict]:
    """Fetch a deduplicated corpus of studies, sepsis-prioritized. Sepsis trials
    are fetched first (so they rank ahead under any stable downstream sort), then
    the remainder is filled with general COMPLETED + results-posted trials up to
    `max_studies`. Each study is cached by NCT id."""
    studies: list[dict] = []
    seen: set[str] = set()

    def _take(batch):
        for s in batch:
            nct = nct_id_of(s)
            if not nct or nct in seen:
                continue
            seen.add(nct)
            studies.append(s)
            if len(studies) >= max_studies:
                return True
        return False

    if sepsis_first:
        for term in SEPSIS_TERMS:
            if len(studies) >= max_studies:
                break
            done = _take(fetch_batch(
                query_term=term, max_studies=max_studies - len(studies),
                require_results=require_results, overall_status=overall_status,
                cache_dir=cache_dir, http_get=http_get))
            if done:
                return studies

    if len(studies) < max_studies:
        _take(fetch_batch(
            max_studies=(max_studies - len(studies)) * 2,  # over-fetch; dedup trims
            require_results=require_results, overall_status=overall_status,
            cache_dir=cache_dir, http_get=http_get))
    return studies[:max_studies]


def iter_specs(studies) -> Iterator:
    """study dict -> TargetTrialSpec (#2) for each study (lazy)."""
    for s in studies:
        yield study_to_spec(s)


def build_spec_corpus(**kwargs):
    """Convenience: fetch the corpus and parse each to a TargetTrialSpec (#2)."""
    return [study_to_spec(s) for s in fetch_corpus(**kwargs)]
