"""#124 shared pre-pass: scan the big lab/measurement tables ONCE, cache, slice per
trial — so lab covariates extract at scale without per-trial rescans. Hermetic:
tiny real-shaped gz fixtures + a tmp cache. Needs pandas + a parquet engine."""

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from tteEngine.adapters import live_loader, mimic, prepass as P
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _gz(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def _mimic_root(tmp):
    root, hosp, icu = tmp / "m", tmp / "m" / "hosp", tmp / "m" / "icu"
    _gz(pd.DataFrame({"hadm_id": [1, 2], "admittime": ["2150-01-01 00:00:00"] * 2,
                      "deathtime": ["", ""]}), hosp / "admissions.csv.gz")
    _gz(pd.DataFrame({"hadm_id": [1, 2], "icd_code": ["99592", "E119"], "icd_version": [9, 10]}),
        hosp / "diagnoses_icd.csv.gz")
    _gz(pd.DataFrame({"itemid": [50813, 50912], "label": ["Lactate", "Creatinine"]}),
        hosp / "d_labitems.csv.gz")
    # labevents for BOTH admissions + both itemids (the corpus-union the pre-pass scans once)
    _gz(pd.DataFrame({"hadm_id": [1, 1, 2], "itemid": [50813, 50912, 50813],
                      "charttime": ["2150-01-01 02:00:00"] * 3, "valuenum": [2.1, 1.0, 3.3]}),
        hosp / "labevents.csv.gz")
    return root


def _index():
    return {"dataset": "MIMIC-IV", "categories": {"lab": [
        {"code": "50813", "name": "Lactate"}, {"code": "50912", "name": "Creatinine"}]}}


def test_prepass_itemids_union_from_index():
    ids = P.prepass_itemids({"lactate", "creatinine"}, _index(), "lab")
    assert ids == {"50813", "50912"}


def test_build_and_slice(tmp_path):
    pp = P.build_mimic_prepass(lab_itemids={"50813", "50912"}, root=str(_mimic_root(tmp_path)),
                               cache_dir=tmp_path / "pp")
    # scanned ONCE -> holds both admissions' rows; slice to a cohort
    assert set(pp.tables["labevents"]["hadm_id"]) == {1, 2}
    sliced = pp.slice({1})
    assert set(sliced["labevents"]["hadm_id"]) == {1}
    assert "label" in sliced["labevents"].columns          # label joined in the pre-pass


def test_cache_roundtrip(tmp_path):
    root, cache = str(_mimic_root(tmp_path)), tmp_path / "pp"
    P.build_mimic_prepass(lab_itemids={"50813"}, root=root, cache_dir=cache)
    assert (cache / "mimic_labevents.parquet").exists()
    pp2 = P.build_mimic_prepass(lab_itemids={"50813"}, root="/nonexistent", cache_dir=cache)  # served from cache
    assert len(pp2.tables["labevents"]) >= 1


def test_load_mimic_prepass_matches_direct_scan(tmp_path):
    root = _mimic_root(tmp_path)
    plan = ExtractionPlan(
        nct_id="N", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="lactate", event_type=EventType.LAB, role="covariate")],
        window_hours=(-48.0, 24.0))

    def resolve(c):
        return {"sepsis": {"99592"}, "lactate": {"Lactate"}}.get(c, {c})

    pp = P.build_mimic_prepass(lab_itemids={"50813", "50912"}, root=str(root), cache_dir=tmp_path / "pp")
    direct = mimic.extract(plan, live_loader.load_mimic(plan, root=str(root), resolve=resolve), resolve=resolve)
    viapp = mimic.extract(plan, live_loader.load_mimic(plan, root=str(root), resolve=resolve, prepass=pp), resolve=resolve)
    # the pre-pass path yields the SAME canonical lab events as the per-trial scan
    d = direct[direct["EVENT_TYPE"] == "lab"].reset_index(drop=True)
    v = viapp[viapp["EVENT_TYPE"] == "lab"].reset_index(drop=True)
    assert list(d["EVENT_VALUE"]) == list(v["EVENT_VALUE"]) == ["2.1"]   # cohort=hadm1, lactate only
    assert set(viapp["TRAJECTORY_ID"]) == {1}


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
