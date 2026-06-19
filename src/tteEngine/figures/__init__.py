"""figures: journal-quality figures from the persisted corpus (#60).

Pure plot-ready data functions (import-light, shared with the #49 UI) + matplotlib
renderers (lazy-imported, `viz` extra). Reads the corpus via
contracts.io.load_comparisons_jsonl; computes nothing statistical (pooling=#64,
calibration metric=#41) — it visualizes.
"""

from .calibration import calibration_plot, calibration_points
from .forest import ForestRow, forest_plot, forest_rows

__all__ = [
    "ForestRow",
    "forest_rows",
    "forest_plot",
    "calibration_points",
    "calibration_plot",
]
