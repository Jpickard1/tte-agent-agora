"""ctgov trial reader + local cache (#1, probe / lane:analysis).

Fetch a ClinicalTrials.gov study (API v2) by NCT id, or a batch by filter, and
cache the raw study JSON locally so every downstream step replays offline and
reproducibly. The cached study carries ``protocolSection`` (eligibility, arms,
interventions, outcomes) and ``resultsSection`` (``outcomeMeasures`` = the trial's
reported effect, the target for the emulated-vs-observed benchmark, #11).

Generalizes ``trialsim/fetch_ctgov.py`` (API-v2 query + JSON) into a typed,
cached, single-study + batch reader. ``requests`` is imported lazily so the
package (and tests using a stub/cache) import without the ``ctgov`` extra.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

API_V2 = "https://clinicaltrials.gov/api/v2"
DEFAULT_CACHE = Path.home() / ".cache" / "tteEngine" / "ctgov"

# An injectable HTTP getter: (url, params) -> parsed JSON dict. Real impl uses
# requests; tests pass a stub so no network is touched.
HttpGet = Callable[[str, "dict | None"], dict]


def _default_get(url: str, params: dict | None = None) -> dict:
    import requests  # lazy — only needed for live fetches

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _cache_path(cache_dir: Path, nct_id: str) -> Path:
    return Path(cache_dir) / f"{nct_id}.json"


def fetch_study(
    nct_id: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    refresh: bool = False,
    http_get: HttpGet | None = None,
) -> dict:
    """Return the full study JSON for one NCT id.

    Cached: once fetched (or whenever the cache file is present) the study is
    served offline unless ``refresh=True``. Idempotent + reproducible.
    """
    nct_id = nct_id.strip().upper()
    cache_dir = Path(cache_dir)
    path = _cache_path(cache_dir, nct_id)
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    get = http_get or _default_get
    study = get(f"{API_V2}/studies/{nct_id}", None)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(study))
    return study


def fetch_batch(
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    page_size: int = 100,
    max_studies: int = 100,
    query_term: str | None = None,
    overall_status: str = "COMPLETED",
    require_results: bool = True,
    http_get: HttpGet | None = None,
) -> list[dict]:
    """Fetch a batch of studies matching a filter (default: COMPLETED +
    results-posted) by paging the API, caching each study by NCT id. Returns the
    list of study dicts (the emulatable corpus for the benchmark, #11)."""
    get = http_get or _default_get
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    params: dict = {
        "pageSize": min(page_size, max_studies),
        "filter.overallStatus": overall_status,
    }
    if require_results:
        params["aggFilters"] = "results:with"  # results-posted only
    if query_term:
        params["query.term"] = query_term

    studies: list[dict] = []
    token: str | None = None
    while len(studies) < max_studies:
        page_params = dict(params)
        if token:
            page_params["pageToken"] = token
        page = get(f"{API_V2}/studies", page_params)
        for study in page.get("studies", []):
            nct = nct_id_of(study)
            if nct:
                _cache_path(cache_dir, nct).write_text(json.dumps(study))
            studies.append(study)
            if len(studies) >= max_studies:
                break
        token = page.get("nextPageToken")
        if not token:
            break
    return studies[:max_studies]


def nct_id_of(study: dict) -> str | None:
    """The NCT id from a study's protocolSection."""
    return (
        study.get("protocolSection", {})
        .get("identificationModule", {})
        .get("nctId")
    )


def reported_outcome_measures(study: dict) -> list[dict]:
    """The resultsSection outcomeMeasures — the trial's REPORTED effect, the
    target the emulated estimate is compared against (#11)."""
    return (
        study.get("resultsSection", {})
        .get("outcomeMeasuresModule", {})
        .get("outcomeMeasures", [])
    )
