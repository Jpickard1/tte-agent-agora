"""#122 data-driven emulability: with a #109 vocab-index dict, a trial is emulable
in a dataset only if its cohort concept resolves to >=1 REAL code there — closing
the emulable-but-empty-cohort gap. Pure (index is a plain dict) -> runs in CI [dev];
no pandas, no real data."""

from tteEngine import triage
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)

# a minimal #109-shaped index: only sepsis codes exist in this (fake) dataset
INDEX = {"dataset": "MIMIC-IV", "categories": {"diagnosis": [
    {"code": "99592", "name": "Severe sepsis", "count": 5257},
    {"code": "R6521", "name": "Severe sepsis with septic shock", "count": 5599},
    {"code": "A419", "name": "Sepsis, unspecified organism", "count": 7770},
]}}


def _trial(nct, condition):
    return TargetTrialSpec(
        nct_id=nct, condition=condition,
        eligibility=[EligibilityCriterion(concept=condition, event_type=EventType.DIAGNOSIS,
                                          comparator=Comparator.EXISTS)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="28-day mortality", concept="death")])


def test_structural_score_unchanged_without_index():
    s = triage.score_spec(_trial("NCT-S", "Sepsis"), "MIMIC-IV")
    assert s.emulable is True
    assert s.condition_resolves is None and s.n_condition_codes is None   # backward-compatible


def test_condition_resolving_to_codes_is_emulable():
    s = triage.score_spec(_trial("NCT-S", "Sepsis"), "MIMIC-IV", index=INDEX)
    assert s.condition_resolves is True and s.n_condition_codes == 3       # 3 sepsis codes
    assert s.emulable is True


def test_condition_with_no_codes_is_not_emulable():
    # a trial structurally emulable but whose condition resolves to ZERO real codes
    s = triage.score_spec(_trial("NCT-RARE", "Fabry disease"), "MIMIC-IV", index=INDEX)
    assert s.condition_resolves is False and s.n_condition_codes == 0
    assert s.emulable is False
    assert any("resolve to no diagnosis codes" in r for r in s.reasons)


def test_token_fallback_for_punctuated_condition():
    # 'Sepsis-3' doesn't substring-match a title, but its 'sepsis' token does
    s = triage.score_spec(_trial("NCT-S3", "Sepsis-3"), "MIMIC-IV", index=INDEX)
    assert s.condition_resolves is True and s.n_condition_codes == 3


def test_build_catalog_data_driven_prunes_unbuildable():
    specs = [_trial("NCT-SEP", "Sepsis"), _trial("NCT-RARE", "Fabry disease")]
    cat = triage.build_catalog(specs, datasets=("MIMIC-IV",), indexes={"MIMIC-IV": INDEX})
    s = cat["summary"]
    assert s["data_driven"] is True
    assert s["n_emulable"] == 1                      # only the sepsis trial is buildable here
    assert s["n_condition_unresolved"] == 1          # the rare-disease trial ruled out, not hidden
    rare = next(r for r in cat["catalog"] if r["nct_id"] == "NCT-RARE")
    assert rare["emulable"] is False and rare["n_condition_codes"] == 0


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
