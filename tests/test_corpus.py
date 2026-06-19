"""#36 corpus batch runner: streaming, no-silent-caps drop logging, and
benchmark aggregation. Synthetic jobs/extract + a stub compare keep it decoupled
from ctgov JSON parsing. Skips without pandas.
"""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import sepsis_vignette as vig  # noqa: E402

from tteEngine.contracts.results import (  # noqa: E402
    Agreement,
    ComparisonResult,
    EffectMeasure,
    TTEResult,
)
from tteEngine.orchestration.corpus import (  # noqa: E402
    DropLog,
    run_corpus,
    run_corpus_benchmark,
    run_corpus_to_jsonl,
)


def _crude_engine(events, cohort, spec):
    """Reuse the Pipeline's bundled crude provider as the injected engine_fn."""
    from tteEngine.orchestration.pipeline import _crude_rr_engine

    return _crude_rr_engine(events, cohort, spec)


def _stub_compare(study, emulated, *, dataset=None):
    """Decouple from ctgov parsing: judge vs a known observed RR=0.9 by side-of-null."""
    same_side = (emulated.estimate - 1.0) * (0.9 - 1.0) > 0
    return ComparisonResult(
        nct_id=emulated.nct_id, dataset=dataset or emulated.dataset, emulated=emulated,
        observed_estimate=0.9, observed_measure=EffectMeasure.RR,
        agreement=Agreement.CONCORDANT if same_side else Agreement.DISCORDANT,
    )


def _jobs(n):
    spec = vig.demo_spec()
    return [({"nct": f"NCT{i:04d}"}, spec.model_copy(update={"nct_id": f"NCT{i:04d}"})) for i in range(n)]


def _tiny_stream():
    """Minimal eligible cohort (treated + control arm, one outcome) — enough to
    exercise the runner machinery without the cost of a full statistical sample."""
    from datetime import datetime, timedelta, timezone

    from tteEngine.contracts.events import CANONICAL_COLUMNS
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows = []
    tid = 1
    for treated, dies in [(True, True), (True, False), (False, True), (False, False)]:
        rows.append((tid, t0 + timedelta(hours=-1), "diagn", "sepsis", "1"))
        rows.append((tid, t0, "lab", "lactate", "4.0"))
        if treated:
            rows.append((tid, t0 + timedelta(hours=2), "medic", "hydrocortisone", "50"))
        if dies:
            rows.append((tid, t0 + timedelta(hours=120), "outco", "death", "1"))
        tid += 1
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _extract_ok(plan, spec, dataset):
    return _tiny_stream()


def _extract_empty(plan, spec, dataset):
    return None


def test_streams_one_row_per_trial_x_dataset():
    rows = list(run_corpus(_jobs(3), ["MIMIC-IV", "eICU-CRD"],
                           extract_fn=_extract_ok, engine_fn=_crude_engine, compare_fn=_stub_compare))
    assert len(rows) == 6  # 3 trials x 2 datasets
    assert all(isinstance(r, ComparisonResult) for r in rows)


def test_run_corpus_is_lazy_generator():
    import types
    gen = run_corpus(_jobs(2), ["MIMIC-IV"], extract_fn=_extract_ok,
                     engine_fn=_crude_engine, compare_fn=_stub_compare)
    assert isinstance(gen, types.GeneratorType)  # streaming, nothing computed yet


def test_no_silent_caps_drops_are_logged_with_reason():
    drops = DropLog()
    rows = list(run_corpus(_jobs(2), ["MIMIC-IV"], extract_fn=_extract_empty,
                           engine_fn=_crude_engine, compare_fn=_stub_compare, drops=drops))
    assert rows == []                      # nothing emulable...
    assert len(drops) == 2                 # ...but every drop is recorded
    assert drops.by_reason() == {"no extractable events": 2}


def test_engine_error_is_dropped_not_fatal():
    def boom(events, cohort, spec):
        raise RuntimeError("estimator blew up")

    drops = DropLog()
    rows = list(run_corpus(_jobs(1), ["MIMIC-IV"], extract_fn=_extract_ok,
                           engine_fn=boom, compare_fn=_stub_compare, drops=drops))
    assert rows == [] and len(drops) == 1
    assert "estimator blew up" in drops.items[0]["reason"]


def test_benchmark_aggregates_and_reports_drops():
    summary, drops = run_corpus_benchmark(
        _jobs(4), ["MIMIC-IV", "eICU-CRD"],
        extract_fn=_extract_ok, engine_fn=_crude_engine, compare_fn=_stub_compare)
    assert summary["n"] == 8                       # 4 x 2 rows aggregated by streaming
    assert "by_dataset" in summary and set(summary["by_dataset"]) == {"MIMIC-IV", "eICU-CRD"}
    assert summary["n_dropped"] == 0
    assert summary["drops_by_reason"] == {}


def _raw_coded_stream():
    """sepsis dx as ICD 'A41', steroid as code 'C05' — concept-level criteria
    only match once a resolver maps codes -> concepts."""
    from datetime import datetime, timedelta, timezone

    from tteEngine.contracts.events import CANONICAL_COLUMNS
    t0 = datetime(2020, 1, 1, tzinfo=timezone.utc)
    rows, tid = [], 1
    for treated, dies in [(True, True), (True, False), (False, True), (False, False)]:
        rows.append((tid, t0 + timedelta(hours=-1), "diagn", "A41", "1"))      # sepsis (ICD)
        rows.append((tid, t0, "lab", "lactate", "4.0"))
        if treated:
            rows.append((tid, t0 + timedelta(hours=2), "medic", "C05", "50"))  # hydrocortisone (code)
        if dies:
            rows.append((tid, t0 + timedelta(hours=120), "outco", "death", "1"))
        tid += 1
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _code_resolver(name):
    return {"A41": "sepsis", "C05": "hydrocortisone"}.get(name, name)


def test_run_corpus_resolve_bridges_raw_codes():
    job = [({"nct": "NCT1"}, vig.demo_spec().model_copy(update={"nct_id": "NCT1"}))]
    raw = lambda plan, spec, ds: _raw_coded_stream()  # noqa: E731

    # identity (no resolve): raw codes don't match concept-level criteria -> dropped
    d1 = DropLog()
    r1 = list(run_corpus(job, ["MIMIC-IV"], extract_fn=raw, engine_fn=_crude_engine,
                         compare_fn=_stub_compare, drops=d1))
    assert r1 == [] and len(d1) == 1

    # with resolver: codes -> concepts -> eligible + treated arm -> emulable
    d2 = DropLog()
    r2 = list(run_corpus(job, ["MIMIC-IV"], extract_fn=raw, engine_fn=_crude_engine,
                         compare_fn=_stub_compare, resolve=_code_resolver, drops=d2))
    assert len(r2) == 1 and len(d2) == 0


def test_make_cohort_provider_threads_resolver():
    from tteEngine.orchestration import make_cohort_provider
    spec = vig.demo_spec()
    prov = make_cohort_provider(resolve=_code_resolver)
    cohort = prov(_raw_coded_stream(), spec, "MIMIC-IV")
    assert cohort.n_total == 4  # all four raw-coded patients now eligible


def test_run_corpus_to_jsonl_persists_and_reads_back_offline(tmp_path):
    # the live-run -> offline bridge: persist via the import-light contracts I/O,
    # then aggregate from the SAVED file with no re-extraction.
    from tteEngine.analysis import run_benchmark
    from tteEngine.contracts.io import load_comparisons_jsonl

    p = tmp_path / "corpus.jsonl"
    n, drops = run_corpus_to_jsonl(
        _jobs(4), ["MIMIC-IV", "eICU-CRD"], p,
        extract_fn=_extract_ok, engine_fn=_crude_engine, compare_fn=_stub_compare)
    assert n == 8 and len(drops) == 0
    rows = list(load_comparisons_jsonl(p))           # read back via contracts loader
    assert len(rows) == 8
    assert run_benchmark(load_comparisons_jsonl(p))["n"] == 8   # offline aggregate


def test_run_corpus_to_jsonl_imports_no_analysis(tmp_path):
    # orchestration.corpus must persist WITHOUT importing the heavy analysis extra
    import subprocess
    import sys
    code = (
        "import sys; "
        "import tteEngine.orchestration.corpus as c; "
        "c.run_corpus_to_jsonl; "
        "assert 'tteEngine.analysis' not in sys.modules, 'persist path pulled analysis'; "
        "print('import-light OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert "import-light OK" in r.stdout, r.stderr


def test_run_corpus_surfaces_assignment_audit_via_on_audit():
    # #138/#130: on_audit fires an AssignmentAudit per emulated (trial,dataset),
    # assembled from the CohortResult primitives -> live_run persists it.
    from tteEngine.contracts.audit import AssignmentAudit
    from tteEngine.orchestration.corpus import assignment_audit_from_cohort  # noqa: F401
    audits = []
    rows = list(run_corpus(_jobs(2), ["MIMIC-IV"], extract_fn=_extract_ok,
                           engine_fn=_crude_engine, compare_fn=_stub_compare,
                           on_audit=audits.append))
    assert len(rows) == 2 and len(audits) == 2
    a = audits[0]
    assert isinstance(a, AssignmentAudit)
    assert a.n_screened >= a.n_enrolled and a.arms  # CONSORT counts + arms populated
    assert any(e.concept == "sepsis" for e in a.eligibility)  # eligibility decisions surfaced


def test_run_corpus_binds_dataset_into_resolvers():
    # run_corpus iterates datasets, so it BINDS ds into the (dataset-aware) resolvers
    # before build_cohort -> a single measurable_fn/arm_match_fn is correct across
    # datasets (not silently wrong when bound to one). Guards probe's multi-dataset run.
    seen_measurable, seen_match = [], []

    def measurable_fn(crit, dataset):       # 2-arg: (criterion, dataset)
        seen_measurable.append(dataset)
        return True

    def arm_match_fn(name, concepts, dataset):  # 3-arg: (name, concepts, dataset)
        seen_match.append(dataset)
        return (name in concepts, None, "name")

    list(run_corpus(_jobs(1), ["MIMIC-IV", "eICU"], extract_fn=_extract_ok,
                    engine_fn=_crude_engine, compare_fn=_stub_compare,
                    measurable_fn=measurable_fn, arm_match_fn=arm_match_fn))
    # each dataset's cohort saw ITS OWN dataset label in the resolvers
    assert set(seen_measurable) == {"MIMIC-IV", "eICU"}
    assert set(seen_match) <= {"MIMIC-IV", "eICU"} and seen_match  # match fn also dataset-bound
