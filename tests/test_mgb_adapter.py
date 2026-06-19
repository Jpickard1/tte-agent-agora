"""Hermetic tests for the MGB -> 5-col adapter (#8, GATED). Synthetic MGB 5-col
fixture only; asserts spec filtering + that the real Snowflake fetch REFUSES
without explicit human-gated opt-in (never touches real MGB)."""

import pytest

pd = pytest.importorskip("pandas")  # data-layer dep ([analysis] extra); skip if absent (CI [dev])

from tteEngine.adapters import mgb
from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan

T0 = pd.Timestamp("2024-03-01 00:00", tz="UTC")


def _events():
    # MGB already emits canonical 5-col; synthetic fixture for 2 trajectories.
    rows = [
        (1, T0, "diagn", "A41", "A41"),                       # sepsis dx (cohort)
        (1, T0 + pd.Timedelta(hours=2), "lab", "Creatinine", "1.2"),
        (1, T0 + pd.Timedelta(hours=200), "lab", "Creatinine", "9.9"),  # out of window
        (2, T0, "diagn", "E11", "E11"),                       # diabetes (out of cohort)
        (2, T0 + pd.Timedelta(hours=2), "lab", "Creatinine", "5.0"),
    ]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def _plan():
    return ExtractionPlan(
        nct_id="NCTTEST", dataset="MGB",
        cohort_filter_concepts=["A41"],
        concepts=[
            ConceptRequest(concept="A41", event_type=EventType.DIAGNOSIS, role="eligibility"),
            ConceptRequest(concept="Creatinine", event_type=EventType.LAB, role="covariate"),
        ],
        window_hours=(-48.0, 24.0),
    )


def test_canonical_and_cohort_concept_window():
    df = mgb.extract(_plan(), _events())
    validate_canonical(df)
    assert set(df["TRAJECTORY_ID"]) == {1}                 # cohort: only sepsis traj
    labs = df[df["EVENT_TYPE"] == "lab"]
    assert list(labs["EVENT_VALUE"]) == ["1.2"]            # +200h lab dropped by window
    assert "A41" in set(df["EVENT_NAME"])


def test_empty_when_no_cohort():
    plan = _plan(); plan.cohort_filter_concepts = ["Z99"]
    df = mgb.extract(plan, _events())
    assert df.empty and tuple(df.columns) == CANONICAL_COLUMNS


def test_real_fetch_is_gated_off_by_default():
    with pytest.raises(RuntimeError, match="human-gated"):
        mgb.fetch_from_snowflake(_plan())                  # no opt-in, no connection -> refuse


def test_real_fetch_refuses_without_connection_even_with_optin():
    with pytest.raises(RuntimeError, match="human-gated"):
        mgb.fetch_from_snowflake(_plan(), i_acknowledge_mgb_is_human_gated=True, connection=None)


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
