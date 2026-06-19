"""#32 cross-dataset variability explainer: forest rows + heterogeneity (I²/τ²)
+ attribution (cohort / coding-measurability / missingness). Measurability
divergence is demonstrated against MGB (gated -> proxies vitals/mortality
regardless of the eICU adapter's reach), so these tests are stable."""

import pytest

from tteEngine import variability as V
from tteEngine.contracts.events import EventType
from tteEngine.contracts.results import ComparisonResult, EffectMeasure, TTEResult
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)


def _cr(dataset, estimate, lo, hi, n_t, n_c, measure=EffectMeasure.OR):
    return ComparisonResult(
        nct_id="NCT-SEP", dataset=dataset,
        emulated=TTEResult(nct_id="NCT-SEP", dataset=dataset, method="iptw", measure=measure,
                           estimate=estimate, ci_low=lo, ci_high=hi, n_treated=n_t, n_control=n_c),
    )


def _full_spec():
    # has a vitals (MEASUREMENT) eligibility + mortality (OUTCOME) -> MGB proxies both
    return TargetTrialSpec(
        nct_id="NCT-SEP", condition="Septic Shock",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS),
                     EligibilityCriterion(concept="map", event_type=EventType.MEASUREMENT,
                                          comparator=Comparator.LT, value=65.0)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"])],
        outcomes=[OutcomeSpec(name="28-day mortality", concept="death")],
    )


def _simple_spec():
    # only dx/lab/med -> measurable in every dataset (no measurability divergence)
    return TargetTrialSpec(
        nct_id="NCT-SEP", condition="Sepsis",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS),
                     EligibilityCriterion(concept="lactate", event_type=EventType.LAB,
                                          comparator=Comparator.GT, value=2.0)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"])],
    )


def test_forest_rows_and_heterogeneity():
    results = [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300),
               _cr("MGB", 0.95, 0.60, 1.50, 40, 40)]
    rep = V.variability_report(_full_spec(), results)
    assert [r["dataset"] for r in rep["forest"]] == ["MIMIC-IV", "MGB"]
    assert rep["heterogeneity"]["k"] == 2 and rep["heterogeneity"]["i2"] is not None


def test_attribution_flags_cohort_and_measurability():
    results = [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300),
               _cr("MGB", 0.95, 0.60, 1.50, 40, 40)]   # 80 vs 600 -> cohort spread
    attr = V.attribute_variability(_full_spec(), results)
    causes = {c["cause"] for c in attr.causes}
    assert "cohort" in causes                          # materially different cohort sizes
    assert "coding/measurability" in causes            # map + death differ (MIMIC vs MGB proxy)
    assert "MGB" in attr.note and attr.note            # human-readable note populated


def test_no_attributable_divergence_when_comparable():
    # same measurability everywhere + comparable cohorts -> no causes
    results = [_cr("MIMIC-IV", 0.70, 0.55, 0.90, 300, 300),
               _cr("eICU-CRD", 0.72, 0.56, 0.92, 290, 300)]
    attr = V.attribute_variability(_simple_spec(), results)
    assert attr.causes == []
    assert "not attributably divergent" in attr.note


def test_missingness_cause_when_frames_differ():
    pd = pytest.importorskip("pandas")
    frames = {
        "MIMIC-IV": pd.DataFrame({"lactate_max": [1.0, 2.0, 3.0, 4.0]}),          # 0% missing
        "MGB": pd.DataFrame({"lactate_max": [1.0, None, None, None]}),            # 75% missing
    }
    results = [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300),
               _cr("MGB", 0.66, 0.48, 0.90, 280, 300)]
    rep = V.variability_report(_full_spec(), results, frames=frames,
                               feature_columns=["lactate_max"])
    causes = {c["cause"] for c in rep["attribution"]["causes"]}
    assert "missingness" in causes


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
