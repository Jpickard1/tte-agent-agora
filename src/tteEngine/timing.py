"""Timing harmonization helpers (#31) — make a TimingConfig drive every adapter.

`effective_window(plan)` gives the extraction window an adapter should use (the
plan's TimingConfig if present, else the legacy window_hours), and
`harmonize_timestamps(df, timing)` floors the canonical TIMESTAMP to the config's
common precision so MIMIC (sub-minute), eICU (minute offsets) and MGB align to
the same grid. `to_time_zero_rule` bridges the config to the cohort builder's
TimeZeroRule so t0 anchoring uses the same clock + grace.

Import-light: only the (pure) timing contract at module top; pandas is imported
lazily inside harmonize_timestamps, so this module loads in CI's [dev] env.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts.timing import DATASET_NATIVE_PRECISION, TimePrecision, TimingConfig

if TYPE_CHECKING:
    import pandas as pd

#: the default shared timing contract (ICU-admission clock, -48h..24h, minute grid).
DEFAULT_TIMING = TimingConfig()

#: TimePrecision -> pandas floor frequency alias.
_FLOOR_FREQ: dict[TimePrecision, str] = {
    TimePrecision.SECOND: "s", TimePrecision.MINUTE: "min",
    TimePrecision.HOUR: "h", TimePrecision.DAY: "D",
}


def effective_window(plan) -> tuple[float, float]:
    """Extraction window for an adapter: the plan's TimingConfig window when set,
    else the legacy plan.window_hours (back-compat)."""
    timing = getattr(plan, "timing", None)
    return timing.extraction_window_hours if timing is not None else plan.window_hours


def harmonize_timestamps(df: "pd.DataFrame", timing: TimingConfig | None) -> "pd.DataFrame":
    """Floor canonical TIMESTAMP to `timing.precision` so cross-dataset timing is
    on one grid. No-op when timing is None (back-compat) or df has no TIMESTAMP."""
    if timing is None or df is None or "TIMESTAMP" not in getattr(df, "columns", []):
        return df
    import pandas as pd

    out = df.copy()
    out["TIMESTAMP"] = pd.to_datetime(out["TIMESTAMP"], utc=True).dt.floor(_FLOOR_FREQ[timing.precision])
    return out


def precision_warnings(timing: TimingConfig, dataset: str) -> list[str]:
    """Flag if the config asks for FINER precision than `dataset` natively supports
    (e.g. 'second' on eICU, which is minute-only) — surfaced, not silently coerced."""
    order = [TimePrecision.DAY, TimePrecision.HOUR, TimePrecision.MINUTE, TimePrecision.SECOND]
    native = DATASET_NATIVE_PRECISION.get(dataset)
    if native is None:
        return []
    if order.index(timing.precision) > order.index(native):
        return [f"{dataset}: requested precision '{timing.precision.value}' is finer than its "
                f"native '{native.value}'; values are at best {native.value}-resolution."]
    return []


def to_time_zero_rule(timing: TimingConfig):
    """Bridge to the cohort builder's TimeZeroRule (same clock + grace), so t0
    anchoring is driven by the same timing contract as extraction."""
    from .contracts.trial_spec import TimeZeroRule

    return TimeZeroRule(anchor=timing.clock.value, grace_window_hours=timing.grace_window_hours)


__all__ = [
    "DEFAULT_TIMING", "effective_window", "harmonize_timestamps",
    "precision_warnings", "to_time_zero_rule",
]
