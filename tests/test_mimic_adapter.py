"""Hermetic tests for the MIMIC-IV -> 5-col adapter (#6). Synthetic MIMIC-shaped
DataFrames (no real CSVs): assert the adapter is spec-driven (cohort + concept +
window filtering) and emits a canonical stream that passes validate_canonical."""

import pytest

pd = pytest.importorskip("pandas")  # data-layer dep ([analysis] extra); skip if absent (CI [dev])

from tteEngine.adapters import mimic
from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _tables():
    admit = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    admissions = pd.DataFrame({"hadm_id": [1, 2], "admittime": [admit, admit]})
    diagnoses = pd.DataFrame({
        "hadm_id": [1, 2],
        "icd_code": ["A41", "E11"],          # hadm1 = sepsis (in cohort); hadm2 = diabetes (out)
        "charttime": [admit, admit],
    })
    labevents = pd.DataFrame({
        "hadm_id": [1, 1, 2],
        "label": ["Creatinine", "Creatinine", "Creatinine"],
        "valuenum": [1.2, 9.9, 5.0],
        # in-window (+2h), out-of-window (+100h), and an out-of-cohort hadm
        "charttime": [admit + pd.Timedelta(hours=2),
                      admit + pd.Timedelta(hours=100),
                      admit + pd.Timedelta(hours=2)],
    })
    return {"admissions": admissions, "diagnoses_icd": diagnoses, "labevents": labevents}


def _plan():
    return ExtractionPlan(
        nct_id="NCTTEST", dataset="MIMIC-IV",
        cohort_filter_concepts=["A41"],
        concepts=[
            ConceptRequest(concept="A41", event_type=EventType.DIAGNOSIS, role="eligibility"),
            ConceptRequest(concept="Creatinine", event_type=EventType.LAB,
                           role="covariate", require_value=True),
        ],
        window_hours=(-48.0, 24.0),
    )


def test_output_is_canonical():
    df = mimic.extract(_plan(), _tables())
    assert tuple(df.columns) == CANONICAL_COLUMNS
    validate_canonical(df)                       # raises if non-canonical
    assert str(df["TRAJECTORY_ID"].dtype) == "int64"
    assert str(df["TIMESTAMP"].dtype).startswith("datetime64")


def test_cohort_filter_excludes_non_matching_admissions():
    df = mimic.extract(_plan(), _tables())
    assert set(df["TRAJECTORY_ID"]) == {1}       # hadm2 (no sepsis dx) excluded


def test_outcome_death_extracted_from_deathtime_unclipped():
    # death (OUTCOME) comes from admissions.deathtime, not a name-keyed table, and
    # is NOT clipped to the extraction window (post-baseline, e.g. +200h).
    tables = _tables()
    admit = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    tables["admissions"] = pd.DataFrame({
        "hadm_id": [1, 2],
        "admittime": [admit, admit],
        "deathtime": [admit + pd.Timedelta(hours=200), pd.NaT],   # hadm1 dies; hadm2 survives
    })
    plan = ExtractionPlan(
        nct_id="X", cohort_filter_concepts=["A41"],
        concepts=[ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome")],
        window_hours=(-48.0, 24.0),
    )
    df = mimic.extract(plan, tables)
    out = df[df["EVENT_TYPE"] == "outco"]
    assert list(out["EVENT_NAME"]) == ["death"]          # emitted by concept name
    assert set(out["TRAJECTORY_ID"]) == {1}              # only the admission with a deathtime
    validate_canonical(df)


def test_concept_and_window_filtering():
    df = mimic.extract(_plan(), _tables())
    # the sepsis diagnosis is captured
    diag = df[df["EVENT_TYPE"] == "diagn"]
    assert list(diag["EVENT_NAME"]) == ["A41"]
    # only the in-window Creatinine (1.2), not the +100h one (9.9)
    labs = df[df["EVENT_TYPE"] == "lab"]
    assert list(labs["EVENT_VALUE"]) == ["1.2"]


def test_empty_when_no_cohort():
    plan = _plan()
    plan.cohort_filter_concepts = ["Z99"]        # nothing matches
    df = mimic.extract(plan, _tables())
    assert df.empty and tuple(df.columns) == CANONICAL_COLUMNS


def test_resolver_injection_maps_concepts_to_codes():
    # a vocab-like resolver (#5 seam): concept 'sepsis' -> the A41 code
    def resolve(concept):
        return {"sepsis": {"A41"}, "creat": {"Creatinine"}}.get(concept, {concept})
    plan = ExtractionPlan(
        nct_id="X", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility")],
    )
    df = mimic.extract(plan, _tables(), resolve=resolve)
    assert set(df["TRAJECTORY_ID"]) == {1}
    # the dx is emitted (alongside the icu_admission anchor every cohort patient now gets)
    assert list(df[df["EVENT_TYPE"] == "diagn"]["EVENT_NAME"]) == ["A41"]
    assert "icu_admission" in set(df[df["EVENT_TYPE"] == "locat"]["EVENT_NAME"])


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
