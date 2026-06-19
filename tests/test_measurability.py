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


def test_vitals_measurement_measurable_in_icu_dbs_proxy_in_mgb():
    # MEASUREMENT (vitals) is now DIRECT in MIMIC (chartevents) AND eICU (#83 wired
    # vitalPeriodic). MGB (gated passthrough) still only proxies it -> surfaced gap.
    for ds in ("MIMIC-IV", "eICU-CRD"):
        assert _by(M.measurability_report(_spec(), ds), "eligibility", "map").status == M.MEASURABLE
    assert _by(M.measurability_report(_spec(), "MGB"), "eligibility", "map").status == M.PROXY


def test_mortality_outcome_measurable_in_icu_dbs_proxy_in_mgb():
    for ds in ("MIMIC-IV", "eICU-CRD"):
        assert _by(M.measurability_report(_spec(), ds), "outcome", "death").status == M.MEASURABLE
    mgb = _by(M.measurability_report(_spec(), "MGB"), "outcome", "death")
    assert mgb.status == M.PROXY and "discharge status" in mgb.reason


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
    rep = M.measurability_report(_spec(), "MGB")     # gated: still proxies vitals + mortality
    s = rep.summary
    assert s["n_elements"] == len(rep.elements)
    assert s["n_measurable"] + s["n_proxy"] + s["n_unmeasurable"] == s["n_elements"]
    assert s["fully_measurable"] is False
    assert any(g["concept"] == "map" for g in s["gaps"])
    assert any(g["concept"] == "death" for g in s["gaps"])


def test_build_catalog_per_trial_per_db():
    cat = M.build_measurability_catalog([_spec(), _spec()], datasets=("MIMIC-IV", "eICU-CRD", "MGB"))
    assert cat["summary"]["n_reports"] == 6 and cat["summary"]["n_trials"] == 2
    assert cat["summary"]["gap_reasons"]                 # MGB proxies -> gaps tallied, not hidden
    # 5 elements/report (3 eligibility + 1 exposure + 1 outcome) x 6 reports = 30
    assert len(cat["elements"]) == 30
    # both ICU DBs are now fully measurable for this spec (all domains direct)
    icu = [r for r in cat["reports"] if r["dataset"] in ("MIMIC-IV", "eICU-CRD")]
    assert all(r["fully_measurable"] for r in icu)


def test_direct_sets_match_adapters_no_drift():
    pytest.importorskip("pandas")
    from tteEngine.adapters import eicu, mimic
    # DIRECT mirrors each adapter's TABLE_SPEC plus the SPECIAL-CASED domains each
    # emits outside TABLE_SPEC (MIMIC deathtime OUTCOME; eICU vitals + mortality, #83).
    assert M.DATASET_DIRECT["MIMIC-IV"] - {EventType.OUTCOME} == set(mimic.TABLE_SPEC)
    assert (M.DATASET_DIRECT["eICU-CRD"] - {EventType.MEASUREMENT, EventType.OUTCOME}
            == set(eicu.TABLE_SPEC))


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
