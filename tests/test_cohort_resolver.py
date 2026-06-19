"""#5<->#9 seam: build_cohort's injected resolver lets concept-level eligibility/
arms match RAW-coded adapter streams (e.g. ICD 'A41' -> concept 'sepsis'), while
the default (identity) leaves concept-name streams unchanged."""

import pytest

pd = pytest.importorskip("pandas")

from tteEngine import vocab
from tteEngine.cohort import build_cohort
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, TargetTrialSpec,
)

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _raw_coded_events():
    # what a REAL adapter emits: raw ICD code + drug name (NOT concept names)
    rows = [
        (1, T0, "diagn", "A41", "A41"),                       # sepsis ICD code
        (1, T0 + pd.Timedelta(hours=2), "medic", "hydrocortisone", "50"),
    ]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _spec():
    return TargetTrialSpec(
        nct_id="NCT-X", condition="Sepsis",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS,
                                          comparator=Comparator.EXISTS)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
    )


def test_resolver_bridges_raw_codes_to_concepts():
    coh = build_cohort(_raw_coded_events(), _spec(), dataset="MIMIC-IV", resolve=vocab.classify)
    treated = [a for a in coh.arms if a.name == "steroid"]
    assert treated and 1 in treated[0].trajectory_ids   # A41 -> sepsis (eligible) + steroid arm


def test_default_identity_does_not_match_raw_codes():
    # without a resolver, concept 'sepsis' won't match raw 'A41' -> not eligible
    coh = build_cohort(_raw_coded_events(), _spec(), dataset="MIMIC-IV")
    assert coh.n_total == 0


def test_concept_name_stream_unchanged_by_default():
    # the synthetic/concept-name path (EVENT_NAME=='sepsis') still works w/o resolver
    rows = [(1, T0, "diagn", "sepsis", "1"),
            (1, T0 + pd.Timedelta(hours=2), "medic", "hydrocortisone", "50")]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    coh = build_cohort(df, _spec(), dataset="MIMIC-IV")
    treated = [a for a in coh.arms if a.name == "steroid"]
    assert treated and 1 in treated[0].trajectory_ids


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
