"""Hermetic tests for the eICU-CRD -> 5-col adapter (#7). Synthetic eICU frames
(offset-based times): assert spec-driven extraction (cohort + concept + offset
window) and a canonical output, parity with the MIMIC adapter's contract (#6)."""

import pytest

pd = pytest.importorskip("pandas")  # data-layer dep ([analysis] extra); skip if absent (CI [dev])

from tteEngine.adapters import eicu
from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _tables():
    patient = pd.DataFrame({"patientunitstayid": [10, 20]})
    diagnosis = pd.DataFrame({
        "patientunitstayid": [10, 20],
        "icd9code": ["99592", "250.00"],     # stay10 = severe sepsis (cohort); stay20 = diabetes (out)
        "diagnosisstring": ["sepsis|severe", "diabetes"],
        "diagnosisoffset": [60, 60],
    })
    lab = pd.DataFrame({
        "patientunitstayid": [10, 10, 20],
        "labname": ["creatinine", "creatinine", "creatinine"],
        "labresult": [1.2, 9.9, 5.0],
        # in-window (+120m), out-of-window (+6000m=100h), out-of-cohort
        "labresultoffset": [120, 6000, 120],
    })
    return {"patient": patient, "diagnosis": diagnosis, "lab": lab}


def _plan():
    return ExtractionPlan(
        nct_id="NCTTEST", dataset="eICU-CRD",
        cohort_filter_concepts=["99592"],
        concepts=[
            ConceptRequest(concept="99592", event_type=EventType.DIAGNOSIS, role="eligibility"),
            ConceptRequest(concept="creatinine", event_type=EventType.LAB,
                           role="covariate", require_value=True),
        ],
        window_hours=(-48.0, 24.0),
    )


def test_output_is_canonical():
    df = eicu.extract(_plan(), _tables())
    assert tuple(df.columns) == CANONICAL_COLUMNS
    validate_canonical(df)
    assert str(df["TIMESTAMP"].dtype).startswith("datetime64")   # offset -> tz-aware ts


def test_cohort_and_concept_and_window_filtering():
    df = eicu.extract(_plan(), _tables())
    assert set(df["TRAJECTORY_ID"]) == {10}              # stay20 (no sepsis) excluded
    labs = df[df["EVENT_TYPE"] == "lab"]
    assert list(labs["EVENT_VALUE"]) == ["1.2"]           # +6000m one dropped by window
    assert list(df[df["EVENT_TYPE"] == "diagn"]["EVENT_NAME"]) == ["99592"]


def test_offset_becomes_ordered_timestamp():
    df = eicu.extract(_plan(), _tables())
    # the diagnosis (offset 60) precedes the lab (offset 120) for stay 10
    assert df["TIMESTAMP"].is_monotonic_increasing
    assert (df["TIMESTAMP"] >= eicu.EPOCH).all()


def test_empty_when_no_cohort():
    plan = _plan()
    plan.cohort_filter_concepts = ["00000"]
    df = eicu.extract(plan, _tables())
    assert df.empty and tuple(df.columns) == CANONICAL_COLUMNS


def test_resolver_injection():
    def resolve(concept):
        return {"sepsis": {"99592"}}.get(concept, {concept})
    plan = ExtractionPlan(
        nct_id="X", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility")],
    )
    df = eicu.extract(plan, _tables(), resolve=resolve)
    assert set(df["TRAJECTORY_ID"]) == {10}


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
