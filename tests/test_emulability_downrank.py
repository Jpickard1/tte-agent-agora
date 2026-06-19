"""#122/#162-B emulability down-ranks: device/non-drug interventions aren't emulable
in ICU drug data; a trial whose only distinguishing drug is ROUTINE (banana bag) has
ambiguous arms. Pure -> CI [dev]."""

from tteEngine import triage
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import Arm, OutcomeSpec, TargetTrialSpec


def _spec(interventions, nct="NCT-X"):
    return TargetTrialSpec(
        nct_id=nct, condition="Sepsis",
        arms=[Arm(name="t", intervention_concepts=interventions),
              Arm(name="c", is_control=True)],
        outcomes=[OutcomeSpec(name="28d mortality", concept="death")])


def test_device_trial_not_emulable():
    # TORAYMYXIN device trial -> no drug events in MIMIC -> not emulable
    s = triage.score_spec(_spec(["Device: TORAYMYXIN hemoperfusion"]), "MIMIC-IV")
    assert s.exposure_ok == 0.0 and s.emulable is False
    assert any("device/non-drug" in r for r in s.reasons)


def test_routine_only_drug_downranked():
    # a thiamine-only trial: banana-bag drug -> ambiguous arms -> exposure halved
    s = triage.score_spec(_spec(["Drug: Thiamine"]), "MIMIC-IV")
    assert s.exposure_ok == 0.5
    assert any("routine high-prevalence" in r for r in s.reasons)


def test_distinguishing_drug_not_downranked():
    # hydrocortisone is NOT routine -> full exposure (a distinguishing treatment)
    s = triage.score_spec(_spec(["Drug: Hydrocortisone"]), "MIMIC-IV")
    assert s.exposure_ok == 1.0
    assert not any("routine" in r for r in s.reasons)


def test_combo_with_a_distinguishing_component_not_routine_downranked():
    # HAT (thiamine+vitC+hydrocortisone): hydrocortisone is distinguishing -> NOT
    # down-ranked as routine (the over-match is handled by combo arm-strategy #162-A)
    s = triage.score_spec(_spec(["Drug: Thiamine", "Drug: Vitamin C", "Drug: Hydrocortisone"]), "MIMIC-IV")
    assert s.exposure_ok == 1.0
    assert not any("routine" in r for r in s.reasons)


def test_intervention_type_and_strip():
    assert triage._intervention_type("Device: Starling SV") == "device"
    assert triage._intervention_type("Drug: Thiamine") == "drug"
    assert triage._intervention_type("Hydrocortisone") == "drug"   # unprefixed
    assert triage._strip_type("Drug: Vitamin C") == "vitamin c"


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
