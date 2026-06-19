"""Tests for the #4 common-format API: validate_canonical + deterministic
long->wide materialize_wide. Skips if pandas (analysis extra) isn't installed.
"""

from datetime import datetime, timedelta, timezone

import pytest

pd = pytest.importorskip("pandas")

from tteEngine.common_format import (  # noqa: E402
    Aggregation,
    FeatureSpec,
    materialize_wide,
    validate_canonical,
)
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType  # noqa: E402

T0 = datetime(2020, 1, 1, 0, 0, tzinfo=timezone.utc)


def _stream():
    """Two trajectories: lactate labs + a steroid med, at known offsets from T0."""
    rows = [
        # traj 1
        (1, T0 + timedelta(hours=-2), "lab", "lactate", "2.0"),
        (1, T0 + timedelta(hours=1), "lab", "lactate", "4.0"),
        (1, T0 + timedelta(hours=3), "lab", "lactate", "3.0"),
        (1, T0 + timedelta(hours=2), "medic", "hydrocortisone", "50"),
        # traj 2
        (2, T0 + timedelta(hours=0), "lab", "lactate", "1.5"),
        (2, T0 + timedelta(hours=30), "lab", "lactate", "9.9"),  # outside a 0-24h window
    ]
    df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    return df


def test_validate_canonical_ok():
    assert validate_canonical(_stream()) is not None


def test_validate_rejects_bad_columns():
    bad = _stream().rename(columns={"EVENT_VALUE": "VAL"})
    with pytest.raises(ValueError):
        validate_canonical(bad)


def test_validate_rejects_tz_naive():
    df = _stream()
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        validate_canonical(df)


def test_materialize_first_last_max():
    df = _stream()
    feats = [
        FeatureSpec(name="lactate_first", event_type=EventType.LAB, event_name="lactate", agg=Aggregation.FIRST),
        FeatureSpec(name="lactate_last", event_type=EventType.LAB, event_name="lactate", agg=Aggregation.LAST),
        FeatureSpec(name="lactate_max", event_type=EventType.LAB, event_name="lactate", agg=Aggregation.MAX),
        FeatureSpec(name="got_steroid", event_type=EventType.MEDICATION, event_name="hydrocortisone", agg=Aggregation.ANY),
    ]
    wide = materialize_wide(df, feats)
    # deterministic: sorted ids, columns in spec order
    assert list(wide.columns) == ["TRAJECTORY_ID", "lactate_first", "lactate_last", "lactate_max", "got_steroid"]
    assert list(wide["TRAJECTORY_ID"]) == [1, 2]
    r1 = wide.set_index("TRAJECTORY_ID").loc[1]
    assert r1["lactate_first"] == 2.0  # earliest by timestamp (-2h)
    assert r1["lactate_last"] == 3.0   # latest (3h)
    assert r1["lactate_max"] == 4.0
    assert bool(r1["got_steroid"]) is True
    assert bool(wide.set_index("TRAJECTORY_ID").loc[2]["got_steroid"]) is False


def test_materialize_window_relative_to_index_time():
    df = _stream()
    feats = [FeatureSpec(name="lactate_max_0_24h", event_type=EventType.LAB,
                         event_name="lactate", agg=Aggregation.MAX, window_hours=(0.0, 24.0))]
    # both trajectories indexed at T0
    wide = materialize_wide(df, feats, index_times={1: T0, 2: T0})
    w = wide.set_index("TRAJECTORY_ID")
    assert w.loc[1]["lactate_max_0_24h"] == 4.0   # -2h excluded; 1h/3h in window
    assert w.loc[2]["lactate_max_0_24h"] == 1.5   # 30h reading excluded by window


def test_materialize_is_deterministic():
    df = _stream()
    feats = [FeatureSpec(name="n_labs", event_type=EventType.LAB, agg=Aggregation.COUNT)]
    a = materialize_wide(df, feats)
    b = materialize_wide(df.sample(frac=1.0, random_state=1), feats)  # shuffled input
    pd.testing.assert_frame_equal(a, b)
