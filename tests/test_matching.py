"""#129 code-based matching: concept -> validated CODE SET (ICD hierarchy + drug
ingredient code sets), match patients by CODE with a confidence tier + matched
code (provenance for #130). Resolver is pure; the catalog scan guards pandas."""

import pytest

from tteEngine import matching as M


# --- conditions: ICD families on the structured icd_code (not title substring) ---

def test_icd_family_matches_dotted_and_undotted():
    cs = M.condition_codeset("Sepsis")
    for code in ("A41.9", "A419", "A410", "R65.21", "R6521", "99592", "0389", "038.9"):
        assert cs.match(code) is not None, code
    assert cs.match("E119") is None and cs.match("I10") is None      # not sepsis


def test_icd_match_carries_provenance_and_method():
    m = M.condition_codeset("Sepsis").match("A4101")
    assert m.method == M.ICD_HIERARCHY and m.concept == "sepsis"
    assert m.low_confidence is False


def test_septic_shock_keyword_maps_to_family():
    cs = M.condition_codeset("Septic shock, refractory")
    assert cs.match("R6521") is not None


def test_unknown_condition_has_no_family():
    assert M.condition_codeset("Atrial fibrillation") is None


# --- drugs: ingredient/brand -> CODE set, match by code, exclude wrong route ---

def _mimic_drug_catalog():
    return [
        {"name": "Solu-Cortef (Hydrocortisone Sod Succ)", "gsn": "004250", "ndc": "00009-0825-01"},
        {"name": "Hydrocortisone", "gsn": "001825", "ndc": "00054-3727-31"},
        {"name": "Hydrocortisone Cream 1%", "gsn": "009999", "ndc": "00168-0044-31"},  # topical -> excluded
        {"name": "Aspirin", "gsn": "004400", "ndc": "00904-2013-61"},
    ]


def test_drug_codeset_rolls_up_brand_and_generic_by_ingredient():
    cs = M.drug_codeset("hydrocortisone", _mimic_drug_catalog(), code_fields=("gsn", "ndc"))
    # brand (Solu-Cortef) + generic both included via the ingredient layer
    assert cs.match("004250") is not None and cs.match("001825") is not None
    assert cs.match("00009-0825-01") is not None
    # matched by CODE, with the ingredient method recorded
    m = cs.match("004250")
    assert m.method == M.INGREDIENT and m.concept == "hydrocortisone"


def test_drug_codeset_excludes_wrong_route_and_other_drugs():
    cs = M.drug_codeset("hydrocortisone", _mimic_drug_catalog())
    assert cs.match("009999") is None      # topical cream excluded
    assert cs.match("004400") is None      # aspirin not an ingredient match


def test_corticosteroid_class_concept_rolls_up_members():
    cat = [{"name": "Methylprednisolone Sod Succ", "gsn": "007777", "ndc": "x"},
           {"name": "Dexamethasone", "gsn": "008888", "ndc": "y"}]
    cs = M.drug_codeset("corticosteroid", cat)
    assert cs.match("007777") is not None and cs.match("008888") is not None


def test_low_confidence_flag():
    low = M.CodeMatch(code="x", name="x", method=M.SUBSTRING, concept="c")
    assert low.low_confidence is True
    assert M.TIER_RANK[M.RXNORM] > M.TIER_RANK[M.SUBSTRING]


def test_eicu_codeset_uses_hicl_field():
    cat = [{"name": "norepinephrine", "drughiclseqno": "001844"},
           {"name": "aspirin", "drughiclseqno": "000003"}]
    cs = M.drug_codeset("norepinephrine", cat, code_fields=("drughiclseqno",))
    assert cs.match("001844") is not None and cs.match("000003") is None


def test_build_drug_catalog_scans_codes(tmp_path):
    pd = pytest.importorskip("pandas")
    root = tmp_path / "m"
    (root / "hosp").mkdir(parents=True)
    pd.DataFrame({"subject_id": [1, 1], "drug": ["Hydrocortisone", "Aspirin"],
                  "gsn": ["001825", "004400"], "ndc": ["a", "b"]}).to_csv(
        root / "hosp" / "prescriptions.csv.gz", index=False, compression="gzip")
    cat = M.build_drug_catalog("MIMIC-IV", root=str(root))
    names = {r["name"] for r in cat}
    assert {"Hydrocortisone", "Aspirin"} <= names
    cs = M.drug_codeset("hydrocortisone", cat)
    assert cs.match("001825") is not None      # code from the scanned catalog


def test_emit_match_provenance_into_canonical_schema():
    from tteEngine.contracts.audit import Confidence, MatchProvenance
    m = M.condition_codeset("Sepsis").match("A4101")
    mp = M.to_match_provenance(m, trajectory_id=7, arm="treated", t_rel_hours=2.0,
                               source_table=M.SOURCE_TABLES[("diagnosis", "MIMIC-IV")])
    assert isinstance(mp, MatchProvenance)
    assert mp.method == Confidence.ICD_HIERARCHY and mp.matched_code == "A4101"
    assert mp.trajectory_id == 7 and mp.arm == "treated" and mp.source_table == "diagnoses_icd"


def test_drug_match_emits_ingredient_provenance():
    from tteEngine.contracts.audit import Confidence
    cs = M.drug_codeset("hydrocortisone", _mimic_drug_catalog())
    mp = M.to_match_provenance(cs.match("004250"), trajectory_id=1, arm="steroid")
    assert mp.method == Confidence.INGREDIENT and mp.concept == "hydrocortisone"


def test_build_drug_matcher_from_spec_and_catalog():
    from tteEngine.contracts.trial_spec import Arm, TargetTrialSpec
    spec = TargetTrialSpec(nct_id="N", arms=[
        Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
        Arm(name="control", is_control=True)])
    matcher = M.build_drug_matcher(spec, "MIMIC-IV", catalog=_mimic_drug_catalog())
    assert set(matcher) == {"hydrocortisone"}                       # arm interventions only
    assert matcher["hydrocortisone"].match("004250") is not None    # code-set built


def test_build_drug_catalog_cache_roundtrip(tmp_path):
    pd = pytest.importorskip("pandas")
    root = tmp_path / "m"
    (root / "hosp").mkdir(parents=True)
    pd.DataFrame({"drug": ["Hydrocortisone"], "gsn": ["001825"], "ndc": ["a"]}).to_csv(
        root / "hosp" / "prescriptions.csv.gz", index=False, compression="gzip")
    cache = tmp_path / "c"
    cat = M.build_drug_catalog("MIMIC-IV", root=str(root), cache_dir=cache)
    assert (cache / "drug_catalog_MIMIC-IV.json").exists()
    again = M.build_drug_catalog("MIMIC-IV", root="/nonexistent", cache_dir=cache)  # served from cache
    assert again == cat


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


# --- #131 fix: ctgov 'Drug: ' prefix + ingredient synonyms (probe's blocker) ---

def _metabolic_catalog():
    # real-MIMIC-shaped names (probe: catalog is healthy; resolution was the bug)
    return [{"name": "Thiamine", "gsn": "004321", "ndc": "n1"},
            {"name": "Hydrocortisone Na Succ", "gsn": "004250", "ndc": "n2"},
            {"name": "Ascorbic Acid", "gsn": "001122", "ndc": "n3"},
            {"name": "Furosemide", "gsn": "008209", "ndc": "n4"}]


def test_drug_prefix_stripped_resolves_codes():
    # 'Drug: Thiamine' must resolve to the catalog 'Thiamine' (was 0 -> regression)
    assert M.drug_codeset("Drug: Thiamine", _metabolic_catalog()).match("004321") is not None
    assert M.drug_codeset("Drug: Hydrocortisone", _metabolic_catalog()).match("004250") is not None


def test_vitamin_c_synonym_to_ascorbic():
    cs = M.drug_codeset("Drug: Vitamin C", _metabolic_catalog())
    assert cs.match("001122") is not None      # Vitamin C -> Ascorbic Acid


def test_clean_intervention_strips_type_prefix():
    assert M.clean_intervention("Drug: Thiamine") == "thiamine"
    assert M.clean_intervention("Biological: Anakinra") == "anakinra"
    assert M.clean_intervention("Hydrocortisone") == "hydrocortisone"


def test_build_drug_matcher_with_ctgov_prefixed_arms():
    from tteEngine.contracts.trial_spec import Arm, TargetTrialSpec
    spec = TargetTrialSpec(nct_id="NCT03509350", arms=[
        Arm(name="HAT", intervention_concepts=["Drug: Thiamine", "Drug: Hydrocortisone", "Drug: Vitamin C"]),
        Arm(name="placebo", is_control=True)])
    m = M.build_drug_matcher(spec, "MIMIC-IV", catalog=_metabolic_catalog())
    # every drug arm concept now resolves >=1 code (the blocker was all-zero)
    assert all(len(cs.codes) >= 1 for c, cs in m.items())
    assert m["Drug: Thiamine"].match("004321") is not None
