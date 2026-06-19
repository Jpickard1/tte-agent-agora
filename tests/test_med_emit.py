"""#131 seam A: the adapter code-matches medications by gsn/ndc/HICL and emits
EVENT_NAME=matched arm concept + EVENT_VALUE JSON {raw_name,code,method,dose,
source_table}. Hermetic; needs pandas."""

import json

import pytest

pd = pytest.importorskip("pandas")

from tteEngine import matching as M
from tteEngine.adapters import eicu, mimic
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _resolve(c):
    return {"sepsis": {"99592"}}.get(c, {c})


def test_mimic_medication_emitted_by_code_as_concept_with_json():
    admit = pd.Timestamp("2150-01-01 00:00", tz="UTC")
    tables = {
        "admissions": pd.DataFrame({"hadm_id": [1], "admittime": [admit], "deathtime": [pd.NaT]}),
        "diagnoses_icd": pd.DataFrame({"hadm_id": [1], "icd_code": ["99592"],
                                       "charttime": [admit]}),
        "prescriptions": pd.DataFrame({
            "hadm_id": [1, 1], "drug": ["Solu-Cortef (Hydrocortisone)", "Aspirin"],
            "starttime": [admit + pd.Timedelta(hours=2)] * 2, "dose_val_rx": ["100", "81"],
            "gsn": ["004250", "004400"], "ndc": ["x", "y"]}),
    }
    plan = ExtractionPlan(
        nct_id="N", cohort_filter_concepts=["sepsis"], window_hours=(-48.0, 24.0),
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="hydrocortisone", event_type=EventType.MEDICATION, role="exposure")])
    matcher = {"hydrocortisone": M.drug_codeset(
        "hydrocortisone", [{"name": "Solu-Cortef (Hydrocortisone)", "gsn": "004250", "ndc": "x"}])}

    df = mimic.extract(plan, tables, resolve=_resolve, drug_matcher=matcher)
    med = df[df["EVENT_TYPE"] == "medic"]
    assert list(med["EVENT_NAME"]) == ["hydrocortisone"]          # the matched ARM CONCEPT, not the drug name
    val = json.loads(med["EVENT_VALUE"].iloc[0])
    assert val["code"] == "004250" and val["method"] == M.INGREDIENT
    assert val["raw_name"] == "Solu-Cortef (Hydrocortisone)" and val["source_table"] == "prescriptions"
    assert val["dose"] == "100"
    # aspirin (no code match) is NOT emitted as the steroid concept
    assert set(med["TRAJECTORY_ID"]) == {1} and len(med) == 1


def test_provenance_from_event_value_roundtrip():
    from tteEngine.contracts.audit import Confidence
    ev = M.med_event_value(raw_name="Solu-Cortef", code="004250", method=M.INGREDIENT,
                           dose="100", source_table="prescriptions")
    mp = M.provenance_from_event_value(ev, trajectory_id=1, arm="steroid", concept="hydrocortisone",
                                       t_rel_hours=2.0)
    assert mp.matched_code == "004250" and mp.matched_event_name == "Solu-Cortef"
    assert mp.method == Confidence.INGREDIENT and mp.concept == "hydrocortisone" and mp.arm == "steroid"


def test_eicu_medication_emitted_by_code_as_concept():
    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [10], "unitdischargestatus": ["Alive"],
                                 "unitdischargeoffset": [2000], "hospitaldischargestatus": ["Alive"],
                                 "hospitaldischargeoffset": [3000]}),
        "diagnosis": pd.DataFrame({"patientunitstayid": [10], "diagnosisoffset": [0],
                                   "icd9code": ["995.92"], "diagnosisstring": ["sepsis"]}),
        "medication": pd.DataFrame({"patientunitstayid": [10, 10], "drugstartoffset": [30, 30],
                                    "drugname": ["NOREPINEPHRINE", "ASPIRIN"], "dosage": ["5", "81"],
                                    "drughiclseqno": ["001844", "000003"]}),
    }
    plan = ExtractionPlan(
        nct_id="N", cohort_filter_concepts=["sepsis"], window_hours=(-48.0, 24.0),
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="norepinephrine", event_type=EventType.MEDICATION, role="exposure")])

    def resolve(c):
        return {"sepsis": {"995.92"}}.get(c, {c})

    matcher = {"norepinephrine": M.drug_codeset(
        "norepinephrine", [{"name": "NOREPINEPHRINE", "drughiclseqno": "001844"}],
        code_fields=("drughiclseqno",))}
    df = eicu.extract(plan, tables, resolve=resolve, drug_matcher=matcher)
    med = df[df["EVENT_TYPE"] == "medic"]
    assert list(med["EVENT_NAME"]) == ["norepinephrine"]
    assert json.loads(med["EVENT_VALUE"].iloc[0])["code"] == "001844"


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
