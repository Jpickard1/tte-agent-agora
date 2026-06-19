"""#102 live-run driver: catalog -> run_corpus (extract->cohort->engine) -> persist
corpus.jsonl + context.jsonl + drops + RESULTS_NARRATIVE + summary, no silent caps.

Uses injected jobs (no ctgov network) + a synthetic 5-col extract_fn routed through
the REAL cohort builder, with a crude engine_fn (the driver wiring is under test,
not the estimator — that's covered by test_engine_provider / test_vignette).
Skips without pandas.
"""
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import sepsis_vignette as vig  # noqa: E402

from tteEngine.contracts.context import load_context_jsonl  # noqa: E402
from tteEngine.contracts.io import load_comparisons_jsonl  # noqa: E402
from tteEngine.contracts.results import (  # noqa: E402
    Agreement,
    ComparisonResult,
    EffectMeasure,
)
from tteEngine.live_run import build_emulable_jobs, run_live  # noqa: E402


def _crude_engine(events, cohort, spec):
    from tteEngine.orchestration.pipeline import _crude_rr_engine
    return _crude_rr_engine(events, cohort, spec)


def _stub_compare(study, emulated, *, dataset=None):
    """Stub studies have no posted results; judge vs a known observed RR=0.9."""
    same_side = (emulated.estimate - 1.0) * (0.9 - 1.0) > 0
    return ComparisonResult(
        nct_id=emulated.nct_id, dataset=dataset or emulated.dataset, emulated=emulated,
        observed_estimate=0.9, observed_measure=EffectMeasure.RR,
        agreement=Agreement.CONCORDANT if same_side else Agreement.DISCORDANT,
    )


def _jobs(n):
    spec = vig.demo_spec()
    return [({"nct": f"NCT{i:04d}"}, spec.model_copy(update={"nct_id": f"NCT{i:04d}"}))
            for i in range(n)]


def _extract_ok(plan, spec, dataset):
    return vig.confounded_stream(scale=1)


def _extract_empty(plan, spec, dataset):
    return None


def test_run_live_writes_full_gallery(tmp_path):
    jobs = _jobs(3)
    specs = [s for _, s in jobs]
    summary = run_live(extract_fn=_extract_ok, engine_fn=_crude_engine,
                       compare_fn=_stub_compare, jobs=jobs, specs=specs, out_dir=tmp_path,
                       datasets=("MIMIC-IV", "eICU-CRD"), figures=False)
    # artifacts
    for name in ("corpus.jsonl", "context.jsonl", "RESULTS_NARRATIVE.md",
                 "drops.jsonl", "summary.json"):
        assert (tmp_path / name).exists(), name
    # 3 trials x 2 datasets all emulate cleanly
    corp = list(load_comparisons_jsonl(tmp_path / "corpus.jsonl"))
    ctx = list(load_context_jsonl(tmp_path / "context.jsonl"))
    assert summary["n_comparisons"] == len(corp) == 6
    assert summary["n_dropped"] == 0
    # context joins on the same (nct_id, dataset) key as the corpus
    assert {(c.nct_id, c.dataset) for c in corp} == {(r.nct_id, r.dataset) for r in ctx}
    assert summary["concordance_rate"] is not None


def test_run_live_logs_drops_no_silent_caps(tmp_path):
    jobs = _jobs(2)
    specs = [s for _, s in jobs]
    summary = run_live(extract_fn=_extract_empty, engine_fn=_crude_engine,
                       jobs=jobs, specs=specs, out_dir=tmp_path,
                       datasets=("MIMIC-IV",), figures=False)
    assert summary["n_comparisons"] == 0
    assert summary["n_dropped"] == 2
    assert summary["drops_by_reason"] == {"no extractable events": 2}
    # every drop is on the explicit ledger
    drops = (tmp_path / "drops.jsonl").read_text().strip().splitlines()
    assert len(drops) == 2


def test_build_emulable_jobs_reports_catalog_no_network(monkeypatch):
    """Catalog filters to emulable + reports counts (incl. max_studies) without
    hiding anything. Stub fetch_corpus so no network is touched."""
    import tteEngine.ctgov as ctgov

    spec_studies = [{"nct": f"NCT{i:04d}"} for i in range(4)]
    monkeypatch.setattr(ctgov, "fetch_corpus", lambda **kw: spec_studies)
    # every study parses to an emulable sepsis spec
    monkeypatch.setattr(ctgov, "study_to_spec",
                        lambda s: vig.demo_spec(nct_id=s["nct"]))
    jobs, specs, catalog = build_emulable_jobs(max_studies=10, datasets=("MIMIC-IV", "eICU-CRD"))
    assert catalog["n_fetched"] == 4 and catalog["max_studies"] == 10
    assert catalog["n_emulable"] == len(jobs) == len(specs)
    assert catalog["n_emulable"] + catalog["n_unemulable"] + catalog["n_unparseable"] == 4
