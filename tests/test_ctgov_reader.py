"""Tests for the ctgov reader + cache (#1, probe). No network — an injected
http_get stub stands in for the live CT.gov API."""

from pathlib import Path

from tteEngine.ctgov.reader import (
    fetch_batch,
    fetch_study,
    nct_id_of,
    reported_outcome_measures,
)


def _fake_study(nct: str) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct},
            "eligibilityModule": {"eligibilityCriteria": "Inclusion: Sepsis-3"},
            "armsInterventionsModule": {"armGroups": [{"label": "hydrocortisone"}]},
        },
        "resultsSection": {
            "outcomeMeasuresModule": {
                "outcomeMeasures": [{"title": "28-day mortality"}]
            }
        },
    }


def test_fetch_caches_and_replays_offline(tmp_path: Path):
    calls = []

    def stub(url, params=None):
        calls.append(url)
        return _fake_study("NCT001")

    s1 = fetch_study("nct001", cache_dir=tmp_path, http_get=stub)  # lowercase -> normalized
    assert nct_id_of(s1) == "NCT001"
    assert len(calls) == 1
    assert (tmp_path / "NCT001.json").exists()

    # second call is served from cache — no new HTTP hit
    s2 = fetch_study("NCT001", cache_dir=tmp_path, http_get=stub)
    assert s2 == s1
    assert len(calls) == 1


def test_refresh_forces_refetch(tmp_path: Path):
    calls = []

    def stub(url, params=None):
        calls.append(url)
        return _fake_study("NCT002")

    fetch_study("NCT002", cache_dir=tmp_path, http_get=stub)
    fetch_study("NCT002", cache_dir=tmp_path, http_get=stub, refresh=True)
    assert len(calls) == 2  # refresh bypasses cache


def test_reported_outcome_measures():
    om = reported_outcome_measures(_fake_study("NCT001"))
    assert om and om[0]["title"] == "28-day mortality"


def test_fetch_batch_caches_each(tmp_path: Path):
    page = {
        "studies": [_fake_study("NCT001"), _fake_study("NCT002")],
        "nextPageToken": None,
    }

    def stub(url, params=None):
        return page

    studies = fetch_batch(cache_dir=tmp_path, max_studies=10, http_get=stub)
    assert len(studies) == 2
    assert (tmp_path / "NCT001.json").exists()
    assert (tmp_path / "NCT002.json").exists()


def test_fetch_batch_respects_max(tmp_path: Path):
    page = {"studies": [_fake_study(f"NCT{i:03d}") for i in range(5)], "nextPageToken": None}

    def stub(url, params=None):
        return page

    studies = fetch_batch(cache_dir=tmp_path, max_studies=3, http_get=stub)
    assert len(studies) == 3
