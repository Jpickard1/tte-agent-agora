"""Common-format helpers (#4, tte1): validate the canonical 5-col stream and
materialize deterministic long->wide feature views.

`validate_canonical` is the gate every adapter (#6/#7/#8) output must pass.
`materialize_wide` (in materialize.py) is the deterministic long->wide API the
cohort builder (#9) and adapter QA build on — the wide feature table is a
reproducible VIEW over the canonical long stream, never a separate source.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tteEngine.contracts.events import CANONICAL_COLUMNS, CANONICAL_DTYPES

from .materialize import Aggregation, FeatureSpec, materialize_wide

if TYPE_CHECKING:  # pandas is an optional (analysis) dependency
    import pandas as pd

__all__ = [
    "validate_canonical",
    "materialize_wide",
    "FeatureSpec",
    "Aggregation",
    "CANONICAL_COLUMNS",
    "CANONICAL_DTYPES",
]


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
                if getattr(df[col].dtype, "tz", None) is None:
                    raise ValueError(f"{col}: TIMESTAMP must be tz-aware (UTC), got tz-naive")
            elif want == "int64" and got not in ("int64", "Int64"):
                raise ValueError(f"{col}: expected int64, got {got}")
    return df
