"""#101 plan-targeted real-data live_loader. Hermetic: writes TINY real-SHAPED gz CSVs
(real MIMIC-IV / eICU-CRD column layouts) to a tmp root, then asserts the live_loader
reshape them into the adapters' injected-table contract AND that extract runs
end-to-end. No /ewsc dependency. Needs pandas (data layer)."""

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.adapters import eicu, live_loader, mimic
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _gz(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def _mimic_root(tmp):
    root = tmp / "mimiciv"
    hosp, icu = root / "hosp", root / "icu"
    _gz(pd.DataFrame({"hadm_id": [1, 2], "admittime": ["2150-01-01 00:00:00", "2150-02-01 00:00:00"],
                      "deathtime": ["2150-01-09 00:00:00", ""]}), hosp / "admissions.csv.gz")
    # real diagnoses_icd: NO charttime; single code per row
    _gz(pd.DataFrame({"hadm_id": [1, 2], "icd_code": ["99592", "E119"], "icd_version": [9, 10]}),
        hosp / "diagnoses_icd.csv.gz")
    # real labevents key on itemid; label lives in d_labitems
    _gz(pd.DataFrame({"itemid": [50813, 50912], "label": ["Lactate", "Creatinine"],
                      "fluid": ["Blood", "Blood"], "category": ["Chem", "Chem"]}),
        hosp / "d_labitems.csv.gz")
    _gz(pd.DataFrame({"labevent_id": [1, 2, 3], "subject_id": [1, 1, 2], "hadm_id": [1, 1, 2],
                      "specimen_id": [1, 1, 2], "itemid": [50813, 50813, 50813],
                      "order_provider_id": ["", "", ""], "charttime": ["2150-01-01 02:00:00"] * 3,
                      "storetime": [""] * 3, "value": ["2.1", "9.9", "5.0"], "valuenum": [2.1, 9.9, 5.0],
                      "valueuom": ["mmol/L"] * 3, "ref_range_lower": [0] * 3, "ref_range_upper": [2] * 3,
                      "flag": [""] * 3, "priority": [""] * 3, "comments": [""] * 3}),
        hosp / "labevents.csv.gz")
    _gz(pd.DataFrame({"hadm_id": [1], "drug": ["Hydrocortisone"], "starttime": ["2150-01-01 02:00:00"],
                      "stoptime": [""], "drug_type": ["MAIN"], "dose_val_rx": ["50"]}),
        hosp / "prescriptions.csv.gz")
    _gz(pd.DataFrame({"itemid": [220045], "label": ["Heart Rate"], "abbreviation": ["HR"],
                      "linksto": ["chartevents"], "category": ["Routine Vital Signs"],
                      "unitname": ["bpm"], "param_type": ["Numeric"],
                      "lownormalvalue": [0], "highnormalvalue": [200]}), icu / "d_items.csv.gz")
    _gz(pd.DataFrame({"hadm_id": [1], "stay_id": [10], "itemid": [220045],
                      "charttime": ["2150-01-01 03:00:00"], "value": ["88"], "valuenum": [88.0],
                      "valueuom": ["bpm"], "warning": [0]}), icu / "chartevents.csv.gz")
    return root


def _sepsis_plan():
    return ExtractionPlan(
        nct_id="NCT-SMOKE", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome"),
                  ConceptRequest(concept="lactate", event_type=EventType.LAB, role="covariate")],
        window_hours=(-48.0, 24.0))


def _resolve(concept):
    return {"sepsis": {"99592", "0389"}, "lactate": {"Lactate"}}.get(concept, {concept})


def test_load_mimic_reshapes_to_adapter_contract_and_extracts(tmp_path):
    root = _mimic_root(tmp_path)
    tables = live_loader.load_mimic(_sepsis_plan(), root=str(root), resolve=_resolve)
    # diagnoses_icd gained a derived charttime (real table has none)
    assert "charttime" in tables["diagnoses_icd"].columns
    assert set(tables["diagnoses_icd"]["hadm_id"]) == {1}            # only the sepsis admission
    # labevents gained a `label` column joined from d_labitems, pre-filtered to the cohort
    assert "label" in tables["labevents"].columns
    assert set(tables["labevents"]["label"]) == {"Lactate"} and set(tables["labevents"]["hadm_id"]) == {1}
    # the loaded tables drive the real adapter end-to-end
    df = mimic.extract(_sepsis_plan(), tables, resolve=_resolve)
    assert set(df["TRAJECTORY_ID"]) == {1}
    assert "outco" in set(df["EVENT_TYPE"])                          # death from deathtime
    assert "diagn" in set(df["EVENT_TYPE"])


def _eicu_root(tmp):
    root = tmp / "eicu"
    _gz(pd.DataFrame({"patientunitstayid": [100, 200],
                      "unitdischargestatus": ["Alive", "Alive"], "unitdischargeoffset": [2000, 2500],
                      "hospitaldischargestatus": ["Expired", "Alive"], "hospitaldischargeoffset": [4000, 5000]}),
        root / "patient.csv.gz")
    _gz(pd.DataFrame({"diagnosisid": [1, 2], "patientunitstayid": [100, 200],
                      "activeupondischarge": [True, True], "diagnosisoffset": [60, 60],
                      "diagnosisstring": ["sepsis|severe", "endocrine|diabetes"],
                      "icd9code": ["995.92, A41.9", "250.00"], "diagnosispriority": ["Primary", "Primary"]}),
        root / "diagnosis.csv.gz")
    _gz(pd.DataFrame({"labid": [1, 2], "patientunitstayid": [100, 200], "labresultoffset": [30, 30],
                      "labtypeid": [1, 1], "labname": ["lactate", "lactate"], "labresult": [2.4, 1.0],
                      "labresulttext": ["2.4", "1.0"], "labmeasurenamesystem": ["mmol/L"] * 2,
                      "labmeasurenameinterface": [""] * 2, "labresultrevisedoffset": [30, 30]}),
        root / "lab.csv.gz")
    _gz(pd.DataFrame({"vitalperiodicid": [1, 2], "patientunitstayid": [100, 100],
                      "observationoffset": [15, 6000], "temperature": [37.0, 37.0], "sao2": [98, 98],
                      "heartrate": [110, 90], "respiration": [20, 18], "cvp": [8, 8], "etco2": [40, 40],
                      "systemicsystolic": [90, 110], "systemicdiastolic": [50, 60], "systemicmean": [60, 75],
                      "pasystolic": [0, 0], "padiastolic": [0, 0], "pamean": [0, 0],
                      "st1": [0, 0], "st2": [0, 0], "st3": [0, 0], "icp": [0, 0]}),
        root / "vitalPeriodic.csv.gz")
    return root


def test_load_eicu_targeted_reads_and_extracts(tmp_path):
    root = _eicu_root(tmp_path)
    plan = ExtractionPlan(
        nct_id="NCT-SMOKE", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="map", event_type=EventType.MEASUREMENT, role="covariate"),
                  ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome")],
        window_hours=(-48.0, 24.0))

    def resolve(c):
        return {"sepsis": {"995.92", "A41.9"}}.get(c, {c})

    tables = live_loader.load_eicu(plan, root=str(root), resolve=resolve)
    assert "vitalperiodic" in tables and "patient" in tables
    # diagnosis prefetch keeps the multi-code sepsis row (substring match)
    assert 100 in set(tables["diagnosis"]["patientunitstayid"])
    # vitals are column-targeted to the requested concept's source columns (+ ids/offset)
    assert "systemicmean" in tables["vitalperiodic"].columns
    df = eicu.extract(plan, tables, resolve=resolve)
    assert "measu" in set(df["EVENT_TYPE"]) and "outco" in set(df["EVENT_TYPE"])


def test_make_extract_fn_dispatches_by_dataset(tmp_path):
    mroot, eroot = _mimic_root(tmp_path), _eicu_root(tmp_path)
    fn = live_loader.make_extract_fn(("MIMIC-IV", "eICU-CRD"), resolve=_resolve,
                                 mimic_root=str(mroot), eicu_root=str(eroot))
    df = fn(_sepsis_plan(), None, "MIMIC-IV")
    assert df is not None and set(df["TRAJECTORY_ID"]) == {1}
    assert fn(_sepsis_plan(), None, "MGB") is None          # gated -> drop (no silent cap)
    assert fn(_sepsis_plan(), None, "OTHER") is None        # unknown dataset


def run():
    import tempfile
    from pathlib import Path
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
