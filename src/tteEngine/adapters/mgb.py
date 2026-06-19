"""MGB (Mass General Brigham) Snowflake -> canonical 5-col adapter (#8, worker1).

GATED: build now, DO NOT run against real MGB. Accessing MGB requires human
data-access verification, so the real Snowflake fetch (`fetch_from_snowflake`)
REFUSES unless an explicit opt-in flag + a live connection are passed. The
adapter logic itself is built + unit-tested against a SYNTHETIC fixture.

The MGB Snowflake pipeline already emits the canonical 5-col format natively
(EHR-DE's extraction_v4_v2), so this adapter mainly filters that event stream to
a trial's cohort + requested concepts + window (same ExtractionPlan contract as
#6/#7). Self-contained per jpic: no EHR-DE import; the MGB conventions live here.
"""
from __future__ import annotations

from typing import Callable, Mapping

import pandas as pd

from tteEngine.common_format import validate_canonical
from tteEngine.contracts.events import CANONICAL_COLUMNS, EventType
from tteEngine.contracts.extraction_plan import ExtractionPlan
from tteEngine.timing import effective_window, harmonize_timestamps

Resolver = Callable[[str], set[str]]


def _identity_resolver(concept: str) -> set[str]:
    return {concept}


def _empty_canonical() -> pd.DataFrame:
    df = pd.DataFrame({c: [] for c in CANONICAL_COLUMNS})
    df["TRAJECTORY_ID"] = df["TRAJECTORY_ID"].astype("int64")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True)
    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        df[c] = df[c].astype("object")
    return df


def extract(plan: ExtractionPlan, events: pd.DataFrame, *,
            resolve: Resolver | None = None) -> pd.DataFrame:
    """Filter a canonical MGB 5-col event stream to `plan`'s cohort + concepts +
    window and return a validate_canonical frame.

    `events` is the MGB 5-col stream (TRAJECTORY_ID/TIMESTAMP/EVENT_TYPE/
    EVENT_NAME/EVENT_VALUE) — the Snowflake pipeline emits this natively. Cohort =
    trajectories with a DIAGNOSIS event matching the plan's cohort_filter_concepts;
    window is relative to each trajectory's first event (admission proxy).
    """
    resolve = resolve or _identity_resolver
    if events is None or events.empty:
        return _empty_canonical()
    ev = events.copy()
    ev["TRAJECTORY_ID"] = ev["TRAJECTORY_ID"].astype("int64")
    ev["TIMESTAMP"] = pd.to_datetime(ev["TIMESTAMP"], utc=True)

    # cohort = trajectories with a diagnosis event matching the cohort filter
    if plan.cohort_filter_concepts:
        codes: set[str] = set()
        for c in plan.cohort_filter_concepts:
            codes |= resolve(c)
        dx = ev[(ev["EVENT_TYPE"] == EventType.DIAGNOSIS.value)
                & ev["EVENT_NAME"].astype(str).isin(codes)]
        cohort = set(dx["TRAJECTORY_ID"])
    else:
        cohort = set(ev["TRAJECTORY_ID"])
    if not cohort:
        return _empty_canonical()
    ev = ev[ev["TRAJECTORY_ID"].isin(cohort)]

    # keep the concepts the plan asks for, per (event_type, resolved names)
    wanted_by_type: dict[str, set[str]] = {}
    for req in plan.concepts:
        wanted_by_type.setdefault(req.event_type.value, set()).update(resolve(req.concept))
    keep = ev.apply(
        lambda r: r["EVENT_NAME"] in wanted_by_type.get(r["EVENT_TYPE"], set()),
        axis=1) if wanted_by_type else pd.Series(True, index=ev.index)
    ev = ev[keep]
    if ev.empty:
        return _empty_canonical()

    # window relative to each trajectory's first (admission-proxy) timestamp
    lo, hi = effective_window(plan)
    anchor = ev.groupby("TRAJECTORY_ID")["TIMESTAMP"].transform("min")
    inwin = (ev["TIMESTAMP"] >= anchor + pd.to_timedelta(lo, "h")) \
        & (ev["TIMESTAMP"] <= anchor + pd.to_timedelta(hi, "h"))
    ev = ev[inwin]
    if ev.empty:
        return _empty_canonical()

    for c in ("EVENT_TYPE", "EVENT_NAME", "EVENT_VALUE"):
        ev[c] = ev[c].astype("object")
    ev = ev.sort_values(["TRAJECTORY_ID", "TIMESTAMP"]).reset_index(drop=True)
    ev = harmonize_timestamps(ev, getattr(plan, "timing", None))  # #31 common precision grid
    return validate_canonical(ev[list(CANONICAL_COLUMNS)])


def fetch_from_snowflake(plan: ExtractionPlan, *,
                         i_acknowledge_mgb_is_human_gated: bool = False,
                         connection=None) -> pd.DataFrame:
    """GATED real-data fetch. REFUSES by default: MGB requires human data-access
    verification, so this raises unless the caller explicitly opts in AND supplies
    a live Snowflake connection. Build + test the pipeline via `extract` on a
    synthetic fixture until access is granted; do NOT bypass this guard.
    """
    if not i_acknowledge_mgb_is_human_gated or connection is None:
        raise RuntimeError(
            "MGB Snowflake access is human-gated (#8): refusing to fetch real MGB "
            "data. This is intentional until data access is verified. To run for "
            "real (once authorized), pass i_acknowledge_mgb_is_human_gated=True and "
            "a live Snowflake connection; otherwise build/test via extract() on a "
            "synthetic 5-col fixture."
        )
    # Query plan (built but not exercised here): the MGB Snowflake schema emits the
    # canonical 5-col directly; pull the cohort + concepts, then reuse extract().
    raise NotImplementedError(  # pragma: no cover - reached only with real access
        "Live MGB Snowflake query is intentionally unimplemented until data access "
        "is granted; wire the credentialed query here, then call extract(plan, df)."
    )
