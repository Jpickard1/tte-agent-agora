"""Anchor fix: adapters emit an ICU-ADMISSION LOCATION event per cohort patient at
admission, so the landmark t0 = admission, NOT the earliest event (= death for an
outcome-only/control trajectory -> immortal-excluded -> empty cohort + collapsed
control arms). Hermetic; needs pandas."""

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.adapters import eicu, mimic
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _resolve(c):
    # MIMIC no-dot ICD-9 / eICU dotted ICD-9 -> give both so the cohort matches
    return {"sepsis": {"99592", "995.92"}}.get(c, {c})


def test_mimic_anchor_is_earliest_event_before_outcome():
    admit = pd.Timestamp("2150-01-01", tz="UTC")
    tables = {
        "admissions": pd.DataFrame({"hadm_id": [1], "admittime": [admit],
                                    "deathtime": [admit + pd.Timedelta(hours=200)]}),
        "diagnoses_icd": pd.DataFrame({"hadm_id": [1], "icd_code": ["99592"], "charttime": [admit]}),
    }
    # control-like trajectory: only a death OUTCOME, no meds
    plan = ExtractionPlan(nct_id="N", cohort_filter_concepts=["sepsis"], window_hours=(-48.0, 24.0),
                          concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                                    ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome")])
    df = mimic.extract(plan, tables, resolve=_resolve)
    loc = df[df["EVENT_TYPE"] == "locat"]
    assert list(loc["EVENT_NAME"]) == ["icu_admission"]
    assert loc["TIMESTAMP"].iloc[0] == admit                       # anchor at admittime
    # the anchor is the EARLIEST event -> t0 = admission, the death OUTCOME is after
    first = df.sort_values("TIMESTAMP").iloc[0]
    assert first["EVENT_TYPE"] == "locat"
    death = df[df["EVENT_TYPE"] == "outco"]["TIMESTAMP"].iloc[0]
    assert death > loc["TIMESTAMP"].iloc[0]                         # outcome NOT immortal


def test_eicu_anchor_at_unit_admission_offset0():
    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [10], "unitdischargestatus": ["Expired"],
                                 "unitdischargeoffset": [4000], "hospitaldischargestatus": ["Expired"],
                                 "hospitaldischargeoffset": [4000]}),
        "diagnosis": pd.DataFrame({"patientunitstayid": [10], "diagnosisoffset": [60],
                                   "icd9code": ["995.92"], "diagnosisstring": ["sepsis"]}),
    }
    plan = ExtractionPlan(nct_id="N", cohort_filter_concepts=["sepsis"], window_hours=(-48.0, 24.0),
                          concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                                    ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome")])
    df = eicu.extract(plan, tables, resolve=_resolve)
    loc = df[df["EVENT_TYPE"] == "locat"]
    assert "icu_admission" in set(loc["EVENT_NAME"])
    assert loc["TIMESTAMP"].iloc[0] == eicu.EPOCH                   # unit-admission offset 0
    assert df.sort_values("TIMESTAMP").iloc[0]["EVENT_TYPE"] == "locat"


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
