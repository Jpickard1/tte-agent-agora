"""Tests for the corpus builder (#1 extension -> #35). No network — stubbed http_get."""

from pathlib import Path

from tteEngine.contracts import TargetTrialSpec
from tteEngine.ctgov.corpus import build_spec_corpus, fetch_corpus
from tteEngine.ctgov.reader import nct_id_of


def _study(nct: str) -> dict:
    return {"protocolSection": {"identificationModule": {"nctId": nct, "briefTitle": nct}},
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": []}}}


def _stub(url, params=None):
    # sepsis batches carry query.term; the general fill batch does not
    if (params or {}).get("query.term"):
        return {"studies": [_study("NCT-SEP1"), _study("NCT-SEP2")], "nextPageToken": None}
    return {"studies": [_study("NCT-GEN1"), _study("NCT-SEP1"), _study("NCT-GEN2")],
            "nextPageToken": None}


def test_sepsis_first_and_dedup(tmp_path: Path):
    studies = fetch_corpus(max_studies=10, http_get=_stub, cache_dir=tmp_path)
    ncts = [nct_id_of(s) for s in studies]
    assert ncts[:2] == ["NCT-SEP1", "NCT-SEP2"]   # sepsis-prioritized
    assert ncts.count("NCT-SEP1") == 1            # dedup across batches
    assert "NCT-GEN1" in ncts and "NCT-GEN2" in ncts


def test_respects_max_studies(tmp_path: Path):
    studies = fetch_corpus(max_studies=3, http_get=_stub, cache_dir=tmp_path)
    assert len(studies) == 3
    assert nct_id_of(studies[0]) == "NCT-SEP1"     # cap keeps sepsis first


def test_sepsis_first_false(tmp_path: Path):
    studies = fetch_corpus(max_studies=2, sepsis_first=False, http_get=_stub, cache_dir=tmp_path)
    # no sepsis pre-pass: just the general fill
    assert {nct_id_of(s) for s in studies} <= {"NCT-GEN1", "NCT-SEP1", "NCT-GEN2"}
    assert len(studies) == 2


def test_build_spec_corpus(tmp_path: Path):
    specs = build_spec_corpus(max_studies=2, http_get=_stub, cache_dir=tmp_path)
    assert all(isinstance(s, TargetTrialSpec) for s in specs)
    assert specs[0].nct_id == "NCT-SEP1"
