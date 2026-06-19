"""Deterministic long->wide feature materialization (#4).

Turns the canonical 5-col long event-stream into an analysis-ready WIDE table
(one row per TRAJECTORY_ID, one column per FeatureSpec). This is probe's
must-have #3: the wide cohort/feature table is a reproducible VIEW over the
canonical stream — never a separate source of truth. The cohort builder (#9) and
the adapters' QA all call this; determinism (stable id order, stable column
order, timestamp-ordered aggregation) is the contract.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from tteEngine.contracts.events import EventType

if TYPE_CHECKING:  # pandas is an optional (analysis) dependency
    import pandas as pd


class Aggregation(str, Enum):
    """How to collapse the events matching a feature into one cell."""

    FIRST = "first"   # earliest by TIMESTAMP
    LAST = "last"     # latest by TIMESTAMP
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    SUM = "sum"
    COUNT = "count"   # number of matching events (always defined, >=0)
    ANY = "any"       # bool: did any matching event occur


_NEEDS_NUMERIC = {Aggregation.MIN, Aggregation.MAX, Aggregation.MEAN, Aggregation.SUM}


class FeatureSpec(BaseModel):
    """One wide-table column derived from the event stream."""

    name: str = Field(..., description="Output column name.")
    event_type: EventType
    event_name: str | None = Field(
        None, description="Match EVENT_NAME exactly; None = any name of this type."
    )
    agg: Aggregation = Aggregation.LAST
    window_hours: tuple[float, float] | None = Field(
        None, description="Inclusive window relative to the trajectory's index time, e.g. (-24, 0)."
    )
    numeric: bool = Field(True, description="Parse EVENT_VALUE to float (non-parseable dropped).")


def _aggregate(sub: "pd.DataFrame", f: FeatureSpec) -> Any:
    import pandas as pd

    if f.agg is Aggregation.COUNT:
        return int(len(sub))
    if f.agg is Aggregation.ANY:
        return bool(len(sub) > 0)
    if len(sub) == 0:
        return None

    values = sub["EVENT_VALUE"]
    if f.numeric or f.agg in _NEEDS_NUMERIC:
        nums = pd.to_numeric(values, errors="coerce").dropna()
        if len(nums) == 0:
            return None
        if f.agg is Aggregation.FIRST:
            return float(nums.iloc[0])
        if f.agg is Aggregation.LAST:
            return float(nums.iloc[-1])
        if f.agg is Aggregation.MIN:
            return float(nums.min())
        if f.agg is Aggregation.MAX:
            return float(nums.max())
        if f.agg is Aggregation.MEAN:
            return float(nums.mean())
        if f.agg is Aggregation.SUM:
            return float(nums.sum())
    # non-numeric FIRST/LAST
    if f.agg is Aggregation.FIRST:
        return values.iloc[0]
    if f.agg is Aggregation.LAST:
        return values.iloc[-1]
    raise ValueError(f"unsupported aggregation {f.agg} for non-numeric feature {f.name!r}")


def materialize_wide(
    df: "pd.DataFrame",
    features: list[FeatureSpec],
    *,
    index_times: dict[int, Any] | None = None,
    id_col: str = "TRAJECTORY_ID",
) -> "pd.DataFrame":
    """Deterministic long->wide: one row per id (sorted), one column per feature
    (spec order). `window_hours` filters events to a window around
    ``index_times[id]`` (skipped for ids absent from index_times). Stable output
    for a given (df, features, index_times) so estimands reproduce.
    """
    import pandas as pd

    ids = sorted(int(x) for x in df[id_col].dropna().unique())
    out: dict[str, list[Any]] = {id_col: list(ids)}

    by_id = {tid: g.sort_values("TIMESTAMP") for tid, g in df.groupby(id_col, sort=True)}

    for f in features:
        col: list[Any] = []
        for tid in ids:
            sub = by_id.get(tid)
            if sub is None:
                col.append(0 if f.agg is Aggregation.COUNT else (False if f.agg is Aggregation.ANY else None))
                continue
            sub = sub[sub["EVENT_TYPE"] == f.event_type.value]
            if f.event_name is not None:
                sub = sub[sub["EVENT_NAME"] == f.event_name]
            if f.window_hours is not None and index_times is not None and tid in index_times:
                lo, hi = f.window_hours
                rel = (sub["TIMESTAMP"] - index_times[tid]).dt.total_seconds() / 3600.0
                sub = sub[(rel >= lo) & (rel <= hi)]
            col.append(_aggregate(sub, f))
        out[f.name] = col

    return pd.DataFrame(out, columns=[id_col, *(f.name for f in features)])
