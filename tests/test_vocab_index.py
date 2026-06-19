"""#109 per-dataset vocabulary index. Hermetic: tiny real-shaped dictionary/coded
CSVs in a tmp root -> build the catalog, resolve concept->real codes, cache+reload.
Needs pandas."""

import json

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.adapters import vocab_index as VI


def _gz(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, compression="gzip")


def _mimic_root(tmp):
    root = tmp / "mimiciv"
    hosp, icu = root / "hosp", root / "icu"
    _gz(pd.DataFrame({"icd_code": ["99592", "E119", "A419"],
                      "long_title": ["Severe sepsis", "Type 2 diabetes", "Sepsis, unspecified organism"]}),
        hosp / "d_icd_diagnoses.csv.gz")
    _gz(pd.DataFrame({"icd_code": ["99592", "99592", "E119"]}), hosp / "diagnoses_icd.csv.gz")
    _gz(pd.DataFrame({"itemid": [50813], "label": ["Lactate"]}), hosp / "d_labitems.csv.gz")
    _gz(pd.DataFrame({"itemid": [220045, 220181], "label": ["Heart Rate", "ABP mean"],
                      "linksto": ["chartevents", "chartevents"]}), icu / "d_items.csv.gz")
    _gz(pd.DataFrame({"drug": ["Hydrocortisone", "Hydrocortisone", "Aspirin"]}),
        hosp / "prescriptions.csv.gz")
    return root


def test_build_mimic_catalog_and_resolve(tmp_path):
    idx = VI.build_vocab_index("MIMIC-IV", root=str(_mimic_root(tmp_path)),
                               cache_dir=tmp_path / "cache")
    cats = idx["categories"]
    assert {"diagnosis", "lab", "medication", "vital"} <= set(cats)
    # data-driven concept->real-code: 'sepsis' resolves to BOTH sepsis codes, not E119
    codes = VI.codes_for(idx, "diagnosis", "sepsis")
    assert codes == {"99592", "A419"}
    # frequency captured + sorted (severe sepsis appears twice)
    top = VI.search(idx, "diagnosis", "sepsis")[0]
    assert top["code"] == "99592" and top["count"] == 2
    assert VI.codes_for(idx, "lab", "lactate") == {"50813"}
    assert "Hydrocortisone" in VI.codes_for(idx, "medication", "hydrocortisone")


def test_cache_roundtrip(tmp_path):
    cache = tmp_path / "cache"
    idx = VI.build_vocab_index("MIMIC-IV", root=str(_mimic_root(tmp_path)), cache_dir=cache)
    assert (cache / "MIMIC-IV.json").exists()
    # second call loads from cache (no root needed)
    again = VI.build_vocab_index("MIMIC-IV", cache_dir=cache)
    assert again["categories"]["diagnosis"] == idx["categories"]["diagnosis"]


def test_build_eicu_multicode_diagnosis(tmp_path):
    root = tmp_path / "eicu"
    _gz(pd.DataFrame({"icd9code": ["995.92, A41.9", "250.00"],
                      "diagnosisstring": ["sepsis|severe sepsis", "endocrine|diabetes"]}),
        root / "diagnosis.csv.gz")
    _gz(pd.DataFrame({"labname": ["lactate", "lactate", "creatinine"]}), root / "lab.csv.gz")
    _gz(pd.DataFrame({"drugname": ["norepinephrine", "aspirin"]}), root / "medication.csv.gz")
    idx = VI.build_vocab_index("eICU-CRD", root=str(root), cache_dir=tmp_path / "cache")
    # eICU dotted multi-codes split into tokens; 'sepsis' resolves to BOTH (dotted!)
    assert VI.codes_for(idx, "diagnosis", "sepsis") == {"995.92", "A41.9"}
    assert VI.codes_for(idx, "lab", "lactate") == {"lactate"}
    assert any(e["name"] == "map" for e in idx["categories"]["vital"])   # fixed vital catalog


def test_register_into_vocab(tmp_path):
    from tteEngine import vocab
    idx = VI.build_vocab_index("MIMIC-IV", root=str(_mimic_root(tmp_path)), cache_dir=tmp_path / "cache")
    added = VI.register_into_vocab(idx, "sepsis_mimic", "diagnosis", "sepsis")
    assert added == {"99592", "A419"}
    assert vocab.resolve("sepsis_mimic") == {"99592", "A419"}    # now a vocab concept


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
