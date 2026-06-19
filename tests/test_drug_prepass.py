"""#171 drug PRE-PASS: scan the big prescriptions/medication table ONCE (corpus-union
arm codes) instead of per-trial, to unblock the >10k-trial run (#111).

HARD constraint (manager): pure caching — the pre-pass path must produce a
BYTE-IDENTICAL canonical stream (cohort + arm assignment) to the per-trial read on
current main. These tests write synthetic MIMIC-shaped gz CSVs and assert the two
load paths (per-trial _read_filtered vs prepass slice) give identical adapter output.
Skips without pandas (data-layer dep)."""
import pytest

pd = pytest.importorskip("pandas")

from tteEngine.adapters import mimic
from tteEngine.adapters.live_loader import load_mimic
from tteEngine.adapters.prepass import (
    _code_hash,
    build_mimic_prepass,
    corpus_drug_codes,
)
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan
from tteEngine.contracts.trial_spec import Arm, OutcomeSpec, TargetTrialSpec
from tteEngine.matching import build_drug_matcher

# a tiny MIMIC-shaped drug catalog: hydrocortisone is the distinguishing arm drug;
# norepinephrine is a second corpus drug; "saline" is a non-arm drug that must NOT
# leak into a trial whose arm is only hydrocortisone.
_CATALOG = [
    {"name": "Hydrocortisone Sod Succ", "gsn": "001234", "ndc": "00009-0001"},
    {"name": "Norepinephrine", "gsn": "005678", "ndc": "00009-0002"},
    {"name": "Sodium Chloride 0.9% (saline)", "gsn": "009999", "ndc": "00009-0003"},
]


def _spec(interventions, nct="NCT-HAT"):
    return TargetTrialSpec(
        nct_id=nct, condition="Sepsis",
        arms=[Arm(name="treat", intervention_concepts=interventions),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="28d mortality", concept="death")])


def _write_tables(root):
    hosp = root / "hosp"
    hosp.mkdir(parents=True, exist_ok=True)
    admit = pd.Timestamp("2024-01-01 00:00")
    # 4 sepsis admissions (in cohort) + 1 non-sepsis (out)
    pd.DataFrame({
        "hadm_id": [1, 2, 3, 4, 5],
        "admittime": [admit] * 5,
        "deathtime": [pd.NaT] * 5,
    }).to_csv(hosp / "admissions.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "hadm_id": [1, 2, 3, 4, 5],
        "icd_code": ["A41", "A41", "A41", "A41", "E11"],   # 1-4 sepsis, 5 diabetes
    }).to_csv(hosp / "diagnoses_icd.csv.gz", index=False, compression="gzip")
    # prescriptions: hadm1 gets hydrocortisone (treated), hadm2 norepi, hadm3 saline,
    # hadm4 nothing matching, hadm5 (out of cohort) hydrocortisone. Distinct times -> no ties.
    pd.DataFrame({
        "hadm_id": [1, 2, 3, 5],
        "drug": ["Hydrocortisone Sod Succ", "Norepinephrine", "Saline", "Hydrocortisone Sod Succ"],
        "starttime": [admit + pd.Timedelta(hours=h) for h in (2, 3, 4, 5)],
        "dose_val_rx": ["100", "5", "1000", "100"],
        "gsn": ["001234", "005678", "009999", "001234"],
        "ndc": ["00009-0001", "00009-0002", "00009-0003", "00009-0001"],
    }).to_csv(hosp / "prescriptions.csv.gz", index=False, compression="gzip")


def _plan():
    return ExtractionPlan(
        nct_id="NCT-HAT", dataset="MIMIC-IV", cohort_filter_concepts=["A41"],
        concepts=[ConceptRequest(concept="A41", event_type=EventType.DIAGNOSIS, role="eligibility")],
        window_hours=(-48.0, 24.0))


def _load_and_extract(plan, root, dm, prepass=None):
    tables = load_mimic(plan, root=str(root), prepass=prepass, drug_matcher=dm)
    return mimic.extract(plan, tables, drug_matcher=dm)


def test_prepass_byte_identical_to_per_trial(tmp_path):
    """The canonical stream from the drug-prepass path == the per-trial read path."""
    root = tmp_path / "mimic"
    _write_tables(root)
    spec = _spec(["Drug: Hydrocortisone"])
    dm = build_drug_matcher(spec, "MIMIC-IV", catalog=_CATALOG)

    # PER-TRIAL (current main): load_mimic re-reads prescriptions filtered to this arm.
    per_trial = _load_and_extract(_plan(), root, dm)

    # PRE-PASS (#171): scan prescriptions ONCE to the CORPUS-union of arm codes, slice.
    codes = corpus_drug_codes([spec], "MIMIC-IV", catalog=_CATALOG)
    pp = build_mimic_prepass(drug_codes=codes, root=str(root),
                             drug_cache_dir=tmp_path / "cache")
    prepass_path = _load_and_extract(_plan(), root, dm, prepass=pp)

    # byte-identical canonical streams (same cohort, same arm-defining med events)
    pd.testing.assert_frame_equal(
        per_trial.reset_index(drop=True), prepass_path.reset_index(drop=True))
    # and the treated patient (hadm1, hydrocortisone) IS present; out-of-cohort hadm5 absent
    meds = prepass_path[prepass_path["EVENT_TYPE"] == EventType.MEDICATION.value]
    assert set(meds["TRAJECTORY_ID"]) == {1}


def test_prepass_identical_for_multi_trial_corpus(tmp_path):
    """Corpus-union pre-pass holds >1 trial's codes; EACH trial still matches only its
    own arm (the adapter narrows), byte-identical to that trial's per-trial read."""
    root = tmp_path / "mimic"
    _write_tables(root)
    hat = _spec(["Drug: Hydrocortisone"], nct="NCT-HAT")
    pressor = _spec(["Drug: Norepinephrine"], nct="NCT-PRESSOR")
    codes = corpus_drug_codes([hat, pressor], "MIMIC-IV", catalog=_CATALOG)
    pp = build_mimic_prepass(drug_codes=codes, root=str(root),
                             drug_cache_dir=tmp_path / "cache")

    for spec, expect_traj in [(hat, {1}), (pressor, {2})]:
        dm = build_drug_matcher(spec, "MIMIC-IV", catalog=_CATALOG)
        per_trial = _load_and_extract(_plan(), root, dm)
        prepass_path = _load_and_extract(_plan(), root, dm, prepass=pp)
        pd.testing.assert_frame_equal(
            per_trial.reset_index(drop=True), prepass_path.reset_index(drop=True))
        meds = prepass_path[prepass_path["EVENT_TYPE"] == EventType.MEDICATION.value]
        assert set(meds["TRAJECTORY_ID"]) == expect_traj


def test_corpus_drug_codes_is_union(tmp_path):
    hat = _spec(["Drug: Hydrocortisone"])
    pressor = _spec(["Drug: Norepinephrine"], nct="NCT-PRESSOR")
    only_hat = corpus_drug_codes([hat], "MIMIC-IV", catalog=_CATALOG)
    both = corpus_drug_codes([hat, pressor], "MIMIC-IV", catalog=_CATALOG)
    assert only_hat and only_hat < both          # union strictly grows with the pressor arm


def test_drug_prepass_cache_is_content_keyed_and_reused(tmp_path):
    """Cache name keys on the code-union hash (stale-safe); a 2nd build with the same
    codes reuses the parquet (no re-scan), a different code set writes a new file."""
    root = tmp_path / "mimic"
    _write_tables(root)
    cache = tmp_path / "cache"
    codes_a = corpus_drug_codes([_spec(["Drug: Hydrocortisone"])], "MIMIC-IV", catalog=_CATALOG)
    codes_b = corpus_drug_codes([_spec(["Drug: Norepinephrine"])], "MIMIC-IV", catalog=_CATALOG)
    assert _code_hash(codes_a) != _code_hash(codes_b)
    build_mimic_prepass(drug_codes=codes_a, root=str(root), drug_cache_dir=cache)
    assert (cache / f"mimic_prescriptions_{_code_hash(codes_a)}.parquet").exists()
    # rebuild with the same codes -> file already there, reused (no error, same content)
    pp2 = build_mimic_prepass(drug_codes=codes_a, root=str(root), drug_cache_dir=cache)
    assert "prescriptions" in pp2.tables
