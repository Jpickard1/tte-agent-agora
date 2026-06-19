"""Common-format helpers (#4, tte1): validate the canonical 5-col stream and
materialize deterministic long->wide feature views.

`validate_canonical` is the gate every adapter (#6/#7/#8) output must pass.
`materialize_wide` is probe's must-have #3: a DETERMINISTIC long->wide API so
estimands are reproducible (the wide cohort/feature table is a derived VIEW over
the canonical long stream, never a separate source of truth). The full
implementation lands with the cohort builder (#9); the signature + contract are
fixed here so all lanes code to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tteEngine.contracts.events import CANONICAL_COLUMNS, CANONICAL_DTYPES

if TYPE_CHECKING:  # pandas is an optional (analysis) dependency
    import pandas as pd

__all__ = ["validate_canonical", "materialize_wide", "CANONICAL_COLUMNS", "CANONICAL_DTYPES"]


def validate_canonical(df: "pd.DataFrame", *, strict_dtypes: bool = True) -> "pd.DataFrame":
    """Assert a DataFrame is a valid canonical 5-col event stream; return it.

    Checks exact column set/order and (optionally) dtypes. Raises ValueError on
    violation so a bad adapter output fails loudly at the seam, not downstream.
    """
    cols = tuple(df.columns)
    if cols != CANONICAL_COLUMNS:
        raise ValueError(
            f"canonical schema violation: expected columns {CANONICAL_COLUMNS}, got {cols}"
        )
    if strict_dtypes:
        for col, want in CANONICAL_DTYPES.items():
            got = str(df[col].dtype)
            if want.startswith("datetime64"):
                if not got.startswith("datetime64"):
                    raise ValueError(f"{col}: expected tz-aware datetime, got {got}")
            elif want == "int64" and got not in ("int64", "Int64"):
                raise ValueError(f"{col}: expected int64, got {got}")
    return df


def materialize_wide(
    df: "pd.DataFrame",
    feature_spec: dict[str, Any],
    *,
    index_times: dict[int, Any] | None = None,
) -> "pd.DataFrame":
    """Deterministic long->wide view: one row per TRAJECTORY_ID, columns per
    feature in `feature_spec`. Contract only — implemented in the cohort builder
    (#9). Determinism (stable column order + aggregation) is required so
    estimands reproduce.
    """
    raise NotImplementedError(
        "materialize_wide is the #4<->#9 contract; implemented with the cohort builder (#9)."
    )
