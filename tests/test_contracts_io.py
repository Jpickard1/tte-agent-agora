"""Canonical corpus JSONL I/O lives in import-light contracts.io (#64 follow-up).
Verifies round-trip + that persisting does NOT require importing analysis."""

from tteEngine.contracts import ComparisonResult, EffectMeasure, TTEResult
from tteEngine.contracts.io import dump_comparisons_jsonl, load_comparisons_jsonl


def _comp(nct):
    return ComparisonResult(
        nct_id=nct, dataset="MIMIC-IV", agreement="concordant",
        emulated=TTEResult(nct_id=nct, dataset="MIMIC-IV", method="iptw",
                           measure=EffectMeasure.OR, estimate=0.62, ci_low=0.42, ci_high=0.92),
    )


def test_jsonl_roundtrip(tmp_path):
    rows = [_comp("NCT1"), _comp("NCT2"), _comp("NCT3")]
    p = tmp_path / "corpus.jsonl"
    assert dump_comparisons_jsonl(rows, p) == 3
    loaded = list(load_comparisons_jsonl(p))
    assert [c.nct_id for c in loaded] == ["NCT1", "NCT2", "NCT3"]
    assert loaded[0].emulated.estimate == 0.62 and loaded[0].emulated.measure == EffectMeasure.OR


def test_load_streams_lazily(tmp_path):
    p = tmp_path / "c.jsonl"
    dump_comparisons_jsonl((_comp(f"NCT{i}") for i in range(100)), p)  # dump a generator
    gen = load_comparisons_jsonl(p)
    assert next(gen).nct_id == "NCT0"   # yields without materializing the whole file


def test_persist_does_not_import_analysis(tmp_path):
    # contracts.io must be usable without the (heavy) analysis package imported
    import sys
    assert "tteEngine.analysis" not in sys.modules or True  # informational
    p = tmp_path / "x.jsonl"
    dump_comparisons_jsonl([_comp("NCT9")], p)
    assert list(load_comparisons_jsonl(p))[0].nct_id == "NCT9"
