"""#33 protocol-vs-data measurability: per-element measurable/proxy/unmeasurable
per dataset + gap surfacing. Pure stdlib (no pandas) -> runs in CI [dev]; the
adapter drift-check guards on pandas."""

import pytest

from tteEngine import measurability as M
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)


def _spec():
    return TargetTrialSpec(
        nct_id="NCT-SEP", title="Steroids in septic shock", condition="Septic Shock",
        eligibility=[
            EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS),
            EligibilityCriterion(concept="lactate", event_type=EventType.LAB,
                                 comparator=Comparator.GT, value=2.0),
            EligibilityCriterion(concept="map", event_type=EventType.MEASUREMENT,
                                 comparator=Comparator.LT, value=65.0),
        ],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="28-day mortality", concept="death")],
    )


def _by(rep, kind, concept):
    return next(e for e in rep.elements if e.kind == kind and e.concept == concept)


def test_lab_and_diagnosis_measurable_in_both():
    for ds in ("MIMIC-IV", "eICU-CRD"):
        rep = M.measurability_report(_spec(), ds)
        assert _by(rep, "eligibility", "sepsis").status == M.MEASURABLE
        assert _by(rep, "eligibility", "lactate").status == M.MEASURABLE


def test_vitals_measurement_measurable_mimic_proxy_eicu():
    # MEASUREMENT (vitals) is direct in MIMIC (chartevents) but only a proxy in
    # eICU (vitalPeriodic exists, adapter not wired) -> a real surfaced gap.
    assert _by(M.measurability_report(_spec(), "MIMIC-IV"), "eligibility", "map").status == M.MEASURABLE
    assert _by(M.measurability_report(_spec(), "eICU-CRD"), "eligibility", "map").status == M.PROXY


def test_mortality_outcome_measurable_mimic_proxy_eicu():
    assert _by(M.measurability_report(_spec(), "MIMIC-IV"), "outcome", "death").status == M.MEASURABLE
    eicu = _by(M.measurability_report(_spec(), "eICU-CRD"), "outcome", "death")
    assert eicu.status == M.PROXY and "discharge status" in eicu.reason


def test_exposure_medication_measurable_both():
    for ds in ("MIMIC-IV", "eICU-CRD"):
        assert _by(M.measurability_report(_spec(), ds), "exposure", "hydrocortisone").status == M.MEASURABLE


def test_soft_outcome_unmeasurable_everywhere():
    spec = _spec()
    spec.outcomes = [OutcomeSpec(name="Quality of life (EQ-5D)", concept="qol")]
    for ds in ("MIMIC-IV", "eICU-CRD"):
        out = _by(M.measurability_report(spec, ds), "outcome", "qol")
        assert out.status == M.UNMEASURABLE


def test_summary_counts_and_gaps_surfaced():
    rep = M.measurability_report(_spec(), "eICU-CRD")
    s = rep.summary
    assert s["n_elements"] == len(rep.elements)
    assert s["n_measurable"] + s["n_proxy"] + s["n_unmeasurable"] == s["n_elements"]
    # eICU has gaps (vitals proxy + mortality proxy) -> not fully measurable, gaps listed
    assert s["fully_measurable"] is False
    assert any(g["concept"] == "map" for g in s["gaps"])
    assert any(g["concept"] == "death" for g in s["gaps"])


def test_build_catalog_per_trial_per_db():
    cat = M.build_measurability_catalog([_spec(), _spec()], datasets=("MIMIC-IV", "eICU-CRD"))
    assert cat["summary"]["n_reports"] == 4 and cat["summary"]["n_trials"] == 2
    assert cat["summary"]["gap_reasons"]                 # gaps tallied, not hidden
    # 5 elements/report (3 eligibility + 1 exposure + 1 outcome) x 4 reports = 20
    assert len(cat["elements"]) == 20
    # MIMIC report for this spec is fully measurable (all domains direct); eICU is not
    mimic_reports = [r for r in cat["reports"] if r["dataset"] == "MIMIC-IV"]
    assert all(r["fully_measurable"] for r in mimic_reports)


def test_direct_sets_match_adapters_no_drift():
    pytest.importorskip("pandas")
    from tteEngine.adapters import eicu, mimic
    # DIRECT mirrors each adapter's TABLE_SPEC (+ MIMIC's deathtime OUTCOME)
    assert M.DATASET_DIRECT["MIMIC-IV"] - {EventType.OUTCOME} == set(mimic.TABLE_SPEC)
    assert M.DATASET_DIRECT["eICU-CRD"] == set(eicu.TABLE_SPEC)


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
