"""#132 cohort dx matching by ICD FAMILY (hierarchy on icd_code), not name/enumerated
codes — decides WHO is in the cohort (e.g. sepsis = A40*/A41*/R652*/...). Hermetic;
needs pandas."""

import pytest

pd = pytest.importorskip("pandas")

from tteEngine import matching as M
from tteEngine.adapters import eicu, live_loader, mimic
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def test_build_dx_matcher_maps_condition_to_family():
    dm = M.build_dx_matcher(["Sepsis", "Atrial fibrillation"])
    assert "Sepsis" in dm and "Atrial fibrillation" not in dm     # only families we curate
    assert dm["Sepsis"].match("A4101") is not None                # A41* family


def test_mimic_cohort_by_icd_family_catches_unenumerated_code():
    admit = pd.Timestamp("2150-01-01", tz="UTC")
    tables = {
        "admissions": pd.DataFrame({"hadm_id": [1, 2, 3], "admittime": [admit] * 3,
                                    "deathtime": [pd.NaT] * 3}),
        # A4101 (sepsis, NOT in a curated enumerated set), R6521 (severe sepsis), E119 (not sepsis)
        "diagnoses_icd": pd.DataFrame({"hadm_id": [1, 2, 3], "icd_code": ["A4101", "R6521", "E119"],
                                       "charttime": [admit] * 3}),
    }
    plan = ExtractionPlan(nct_id="N", cohort_filter_concepts=["Sepsis"], window_hours=(-48.0, 24.0),
                          concepts=[ConceptRequest(concept="Sepsis", event_type=EventType.DIAGNOSIS,
                                                   role="eligibility")])
    dxm = M.build_dx_matcher(["Sepsis"])
    df = mimic.extract(plan, tables, resolve=lambda c: {c}, dx_matcher=dxm)
    # family match -> hadm 1 (A4101) + 2 (R6521) in cohort; hadm 3 (diabetes) excluded
    assert set(df["TRAJECTORY_ID"]) == {1, 2}
    dx = df[df["EVENT_TYPE"] == "diagn"]
    assert set(dx["EVENT_NAME"]) == {"A4101", "R6521"}            # emitted via family match


def test_eicu_cohort_by_family_handles_multicode_field():
    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [10, 20], "unitdischargestatus": ["Alive"] * 2,
                                 "unitdischargeoffset": [2000] * 2, "hospitaldischargestatus": ["Alive"] * 2,
                                 "hospitaldischargeoffset": [3000] * 2}),
        "diagnosis": pd.DataFrame({"patientunitstayid": [10, 20], "diagnosisoffset": [0, 0],
                                   "icd9code": ["995.92, A41.9", "250.00"],
                                   "diagnosisstring": ["sepsis", "diabetes"]}),
    }
    plan = ExtractionPlan(nct_id="N", cohort_filter_concepts=["Sepsis"], window_hours=(-48.0, 24.0),
                          concepts=[ConceptRequest(concept="Sepsis", event_type=EventType.DIAGNOSIS,
                                                   role="eligibility")])
    df = eicu.extract(plan, tables, resolve=lambda c: {c}, dx_matcher=M.build_dx_matcher(["Sepsis"]))
    assert set(df["TRAJECTORY_ID"]) == {10}                       # A41.9 token in the family


def test_loader_prefetches_family_codes(tmp_path):
    root, hosp = tmp_path / "m", tmp_path / "m" / "hosp"
    hosp.mkdir(parents=True)
    admit = "2150-01-01 00:00:00"
    pd.DataFrame({"hadm_id": [1, 2], "admittime": [admit, admit], "deathtime": ["", ""]}).to_csv(
        hosp / "admissions.csv.gz", index=False, compression="gzip")
    pd.DataFrame({"hadm_id": [1, 2], "icd_code": ["A4101", "E119"], "icd_version": [10, 10]}).to_csv(
        hosp / "diagnoses_icd.csv.gz", index=False, compression="gzip")
    plan = ExtractionPlan(nct_id="N", cohort_filter_concepts=["Sepsis"], window_hours=(-48.0, 24.0),
                          concepts=[ConceptRequest(concept="Sepsis", event_type=EventType.DIAGNOSIS,
                                                   role="eligibility")])
    tabs = live_loader.load_mimic(plan, root=str(root), resolve=lambda c: {c},
                                  dx_matcher=M.build_dx_matcher(["Sepsis"]))
    # A4101 prefetched by family even though not an enumerated 'Sepsis' code
    assert set(tabs["diagnoses_icd"]["hadm_id"]) == {1}


def run():
    import tempfile
    from pathlib import Path
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
