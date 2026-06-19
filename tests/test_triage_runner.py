"""#35 corpus runner: corpus -> specs -> ranked catalog (CSV+JSON+summary), with
the never-silently-cap invariants jpic mandated. Hermetic — a stubbed http_get
for the live-fetch path, and a committed sample-studies fixture for the
parse->score path. Pure stdlib (no pandas)."""

import csv
import json
from pathlib import Path

from tteEngine.ctgov.reader import nct_id_of
from tteEngine.triage import run_corpus_triage

FIXTURE = Path(__file__).parent / "fixtures" / "sample_studies.json"


def _sample_studies() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def _stub(url, params=None):
    # mirrors the corpus reader's API shape: sepsis batches carry query.term.
    sep, gen = _sample_studies()
    if (params or {}).get("query.term"):
        return {"studies": [sep], "nextPageToken": None}
    return {"studies": [gen, sep], "nextPageToken": None}


def test_fixture_parses_to_emulable_and_kept_unemulable(tmp_path: Path):
    cat = run_corpus_triage(studies=_sample_studies(), out_dir=tmp_path)
    summ = cat["summary"]
    # 2 trials x 2 datasets = 4 rows; NOTHING dropped
    assert summ["n_rows"] == 4 and summ["n_trials"] == 2
    assert summ["n_specs_parsed"] == 2 and summ["n_unparseable"] == 0
    # the sepsis-steroid trial is emulable in both datasets; the QoL trial is kept-but-not
    assert summ["n_emulable"] == 2 and summ["n_not_emulable"] == 2
    assert summ["n_emulable_sepsis"] == 2
    # sepsis sorted first; drop-reasons logged (not hidden)
    assert cat["catalog"][0]["is_sepsis"] is True
    assert summ["not_emulable_reasons"]


def test_writes_csv_json_summary(tmp_path: Path):
    run_corpus_triage(studies=_sample_studies(), max_studies=2, out_dir=tmp_path)
    csv_path, jpath, spath = (tmp_path / f for f in ("catalog.csv", "catalog.json", "summary.json"))
    assert csv_path.exists() and jpath.exists() and spath.exists()
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 4
    assert rows[0]["is_sepsis"] == "True"           # sepsis-first survives CSV round-trip
    assert json.loads(spath.read_text())["cap"]["hit"] is True   # studies==cap is flagged


def test_live_fetch_path_via_stub(tmp_path: Path):
    # exercises fetch_corpus(http_get=stub) -> study_to_spec -> build_catalog end-to-end
    cat = run_corpus_triage(max_studies=10, http_get=_stub, cache_dir=tmp_path / "cache",
                            out_dir=tmp_path / "out")
    ncts = {r["nct_id"] for r in cat["catalog"]}
    assert "NCT-SEP-STEROID" in ncts and "NCT-QOL" in ncts
    assert cat["summary"]["n_studies_fetched"] == 2


def test_cap_flag_false_when_below_ceiling():
    cat = run_corpus_triage(studies=_sample_studies(), max_studies=1000)
    assert cat["summary"]["cap"]["hit"] is False     # 2 << 1000: not capped


def test_unparseable_studies_kept_not_dropped():
    # a malformed study (study_to_spec tolerates it -> empty nct) still never crashes
    # the run; force a hard failure to prove it's recorded, not silently dropped.
    bad = {"protocolSection": None}                  # .get on None -> AttributeError in parser
    cat = run_corpus_triage(studies=[*_sample_studies(), bad])
    summ = cat["summary"]
    assert summ["n_studies_fetched"] == 3
    assert summ["n_unparseable"] == 1 and summ["unparseable"]   # logged, surfaced
    assert summ["n_specs_parsed"] == 2                          # the 2 good ones still scored


def run():
    import tempfile
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        if "tmp_path" in t.__code__.co_varnames[: t.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
        else:
            t()
        print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
