"""#133 lab + demographic matching layer: lab concept -> real code set (MIMIC
itemids / eICU labnames) via the #109 index + synonyms; demographic recognizer.
Pure (index is a dict) -> CI [dev]. (Labs are pruned in the lean run, so the
adapter lab-emit integration trails; this is the reusable resolver for full mode.)"""

from tteEngine import matching as M

MIMIC_LAB_INDEX = {"categories": {"lab": [
    {"code": "50813", "name": "Lactate"},
    {"code": "50912", "name": "Creatinine"},
    {"code": "51265", "name": "Platelet Count"},
    {"code": "51301", "name": "White Blood Cells"}]}}

EICU_LAB_INDEX = {"categories": {"lab": [
    {"code": "lactate", "name": "lactate"},
    {"code": "creatinine", "name": "creatinine"}]}}


def test_build_lab_matcher_mimic_itemids():
    m = M.build_lab_matcher(["lactate", "creatinine", "wbc"], "MIMIC-IV", MIMIC_LAB_INDEX)
    assert m["lactate"] == {"50813"}
    assert m["creatinine"] == {"50912"}
    assert m["wbc"] == {"51301"}            # 'white blood cell' synonym -> itemid


def test_build_lab_matcher_eicu_labnames():
    m = M.build_lab_matcher(["lactate"], "eICU-CRD", EICU_LAB_INDEX)
    assert m["lactate"] == {"lactate"}


def test_lab_concept_absent_from_dataset_omitted():
    m = M.build_lab_matcher(["procalcitonin"], "MIMIC-IV", MIMIC_LAB_INDEX)
    assert "procalcitonin" not in m         # not in the catalog -> no code set


def test_demographic_recognizer():
    assert M.is_demographic("age") and M.is_demographic("Sex") and M.is_demographic("gender")
    assert not M.is_demographic("lactate") and not M.is_demographic("sepsis")


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
