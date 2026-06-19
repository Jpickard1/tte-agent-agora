"""#34 missingness + proxy-substitution: proxy list (from #33), data missingness
summary, and sensitivity-of-estimate-to-proxy. Pure parts run in CI [dev]; the
missingness summary guards on pandas."""

import pytest

from tteEngine import missingness as mz
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)


def _spec():
    return TargetTrialSpec(
        nct_id="NCT-SEP", condition="Septic Shock",
        eligibility=[
            EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS),
            EligibilityCriterion(concept="map", event_type=EventType.MEASUREMENT,
                                 comparator=Comparator.LT, value=65.0),
        ],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"])],
        outcomes=[OutcomeSpec(name="28-day mortality", concept="death")],
    )


def test_proxy_list_from_measurability():
    # MGB (gated) proxies vitals (map) + mortality (death); MIMIC + eICU now
    # extract both directly (#83) -> no proxies.
    mgb = mz.proxy_substitution_list(_spec(), "MGB")
    concepts = {p["concept"] for p in mgb}
    assert {"map", "death"} <= concepts
    assert mz.proxy_substitution_list(_spec(), "MIMIC-IV") == []
    assert mz.proxy_substitution_list(_spec(), "eICU-CRD") == []


def test_proxy_sensitivity_robust_when_same_side():
    s = mz.proxy_sensitivity(0.62, 0.70)        # both < 1.0 -> benefit either way
    assert s["robust_to_proxy"] is True
    assert s["conclusion_full"] == "benefit" and s["conclusion_reduced"] == "benefit"
    assert s["abs_delta"] == 0.08


def test_proxy_sensitivity_flags_conclusion_flip():
    s = mz.proxy_sensitivity(0.62, 1.10)        # benefit -> harm when proxy dropped
    assert s["robust_to_proxy"] is False
    assert s["conclusion_full"] == "benefit" and s["conclusion_reduced"] == "harm"


def test_combined_report_has_three_parts():
    rep = mz.missingness_and_proxy_report(
        _spec(), "MGB", sensitivity=mz.proxy_sensitivity(0.62, 0.65))   # MGB still proxies
    assert rep["n_proxies"] >= 2 and rep["proxy_list"]
    assert rep["proxy_sensitivity"]["robust_to_proxy"] is True
    assert "missingness" not in rep                   # no frame supplied -> omitted


def test_missingness_summary():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({
        "TRAJECTORY_ID": [1, 2, 3, 4],
        "group": ["steroid", "control", "steroid", "control"],
        "time_zero": pd.to_datetime(["2024-01-01"] * 4, utc=True),
        "lactate_max": [2.1, None, 3.4, None],         # 50% missing
        "outcome_death": [False, True, False, True],   # 0% missing
    })
    s = mz.missingness_summary(frame)
    assert s["n_rows"] == 4 and s["n_features"] == 2
    assert s["columns"]["lactate_max"]["missing_fraction"] == 0.5
    assert s["columns"]["outcome_death"]["missing_fraction"] == 0.0
    assert s["worst_feature"] == "lactate_max" and s["mean_missing_fraction"] == 0.25


def test_missingness_reads_build_analysis_frame_output():
    pytest.importorskip("pandas")
    import pandas as pd

    from tteEngine.cohort import build_analysis_frame, build_cohort
    from tteEngine.common_format import Aggregation, FeatureSpec
    from tteEngine.contracts.events import CANONICAL_COLUMNS

    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    rows = [
        (1, t0, "diagn", "sepsis", "1"), (1, t0, "medic", "hydrocortisone", "50"),
        (1, t0, "lab", "lactate", "3.0"),
        (2, t0, "diagn", "sepsis", "1"),  # no lactate lab -> missing covariate
    ]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    spec = _spec()
    spec.eligibility = [EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS)]
    cohort = build_cohort(df, spec, dataset="MIMIC-IV")
    frame = build_analysis_frame(df, cohort, spec,
                                 covariates=[FeatureSpec(name="lactate_max", event_type=EventType.LAB,
                                                         event_name="lactate", agg=Aggregation.MAX)])
    s = mz.missingness_summary(frame, cohort.feature_columns)
    assert s["columns"]["lactate_max"]["missing_fraction"] == 0.5   # traj 2 has no lactate


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
