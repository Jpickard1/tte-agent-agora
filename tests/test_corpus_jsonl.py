"""Corpus JSONL persistence (#36 bridge): write/read round-trip, streaming, and
offline benchmark from the saved file. Skips without pandas."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import sepsis_vignette as vig  # noqa: E402

from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult  # noqa: E402
from tteEngine.orchestration.corpus import (  # noqa: E402
    read_corpus_jsonl,
    run_corpus_to_jsonl,
    write_corpus_jsonl,
)


def _cr(nct, dataset, est, agreement):
    return ComparisonResult(
        nct_id=nct, dataset=dataset,
        emulated=TTEResult(nct_id=nct, dataset=dataset, method="iptw",
                           measure=EffectMeasure.OR, estimate=est, ci_low=est * 0.8,
                           ci_high=est * 1.2, n_treated=10, n_control=10,
                           extra={"p_value": 0.04}),
        observed_estimate=0.9, observed_measure=EffectMeasure.RR, agreement=agreement,
    )


def test_write_read_roundtrip(tmp_path):
    rows = [_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT),
            _cr("NCT2", "eICU-CRD", 1.3, Agreement.DISCORDANT)]
    p = tmp_path / "corpus.jsonl"
    n = write_corpus_jsonl(rows, p)
    assert n == 2
    back = list(read_corpus_jsonl(p))
    assert [c.nct_id for c in back] == ["NCT1", "NCT2"]
    assert back[0].emulated.estimate == 0.6
    assert back[0].emulated.extra["p_value"] == 0.04   # diagnostics survive in .extra
    assert back[1].agreement == Agreement.DISCORDANT


def test_read_is_streaming_generator(tmp_path):
    import types
    p = tmp_path / "c.jsonl"
    write_corpus_jsonl([_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT)], p)
    assert isinstance(read_corpus_jsonl(p), types.GeneratorType)


def test_one_object_per_line(tmp_path):
    p = tmp_path / "c.jsonl"
    write_corpus_jsonl([_cr(f"NCT{i}", "MIMIC-IV", 0.6, Agreement.CONCORDANT) for i in range(5)], p)
    assert p.read_text().count("\n") == 5  # one line per row


def test_run_to_jsonl_then_offline_benchmark(tmp_path):
    from tteEngine.analysis import run_benchmark
    from tteEngine.orchestration.pipeline import _crude_rr_engine

    spec = vig.demo_spec()
    jobs = [({"nct": "NCT1"}, spec.model_copy(update={"nct_id": "NCT1"}))]
    stub = lambda study, emulated, *, dataset=None: _cr(emulated.nct_id, dataset, emulated.estimate, Agreement.DISCORDANT)  # noqa: E731
    p = tmp_path / "corpus.jsonl"
    n, drops = run_corpus_to_jsonl(
        jobs, ["MIMIC-IV", "eICU-CRD"], p,
        extract_fn=lambda *a: vig.confounded_stream(1), engine_fn=_crude_rr_engine, compare_fn=stub)
    assert n == 2 and len(drops) == 0
    # downstream reads the SAVED file (no re-extraction) and aggregates
    summary = run_benchmark(read_corpus_jsonl(p))
    assert summary["n"] == 2
