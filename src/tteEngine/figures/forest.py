"""Forest plot of the emulated-vs-observed corpus (#60).

Two layers, intentionally split so the UI (#49) and the journal figures share one
source of truth without the UI needing matplotlib:

  * `forest_rows(comparisons)` — PURE, import-light: turns a ComparisonResult
    stream (or a contracts.io.load_comparisons_jsonl(path) stream) into plot-ready
    rows. This is what #49 renders.
  * `forest_plot(...)` — renders those rows to an image file via matplotlib
    (lazy-imported, Agg backend; needs the `viz` extra). The journal figure.

Reads the persisted corpus; computes nothing statistical itself (pooling is #64,
calibration is #41) — it visualizes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel

if TYPE_CHECKING:
    from tteEngine.contracts.results import ComparisonResult

#: ratio effect measures share a multiplicative null at 1.0 + a log x-axis.
_RATIO = {"RR", "OR", "HR"}


class ForestRow(BaseModel):
    """One plot-ready forest line."""

    label: str
    dataset: str
    measure: str
    estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    observed_estimate: float | None = None
    agreement: str | None = None


def forest_rows(comparisons: Iterable["ComparisonResult"]) -> list[ForestRow]:
    """Plot-ready rows from the corpus (streams; one row per comparison that has
    a finite point estimate)."""
    rows: list[ForestRow] = []
    for c in comparisons:
        e = c.emulated
        if e.estimate is None or e.estimate != e.estimate:  # skip NaN/missing
            continue
        rows.append(ForestRow(
            label=f"{c.nct_id} [{c.dataset}]",
            dataset=c.dataset,
            measure=e.measure.value if hasattr(e.measure, "value") else str(e.measure),
            estimate=float(e.estimate),
            ci_low=e.ci_low, ci_high=e.ci_high,
            observed_estimate=c.observed_estimate,
            agreement=c.agreement.value if hasattr(c.agreement, "value") else str(c.agreement),
        ))
    return rows


def forest_plot(comparisons, path, *, title: str = "Emulated treatment effects",
                max_rows: int = 60) -> str:
    """Render a horizontal forest plot to `path` (PNG/SVG by extension). Ratio
    measures use a log x-axis with the null at 1.0; otherwise linear at 0.0.
    Marks the trial's observed estimate (x) beside each emulated point. Returns
    the path. Needs the `viz` extra (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")  # headless file output
    import matplotlib.pyplot as plt

    rows = comparisons if isinstance(comparisons, list) and comparisons and isinstance(
        comparisons[0], ForestRow) else forest_rows(comparisons)
    if max_rows and len(rows) > max_rows:
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("no finite-estimate comparisons to plot")

    ratio = sum(r.measure in _RATIO for r in rows) >= len(rows) / 2
    null = 1.0 if ratio else 0.0
    y = list(range(len(rows)))[::-1]  # first row at top

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(rows) + 1)))
    for yi, r in zip(y, rows):
        lo = r.ci_low if r.ci_low is not None else r.estimate
        hi = r.ci_high if r.ci_high is not None else r.estimate
        color = {"concordant": "#1a9850", "discordant": "#d73027"}.get(r.agreement or "", "#4575b4")
        ax.plot([lo, hi], [yi, yi], color=color, lw=1.5, zorder=1)
        ax.plot(r.estimate, yi, "o", color=color, ms=5, zorder=2)
        if r.observed_estimate is not None:
            ax.plot(r.observed_estimate, yi, "x", color="black", ms=6, zorder=3)
    ax.axvline(null, color="grey", ls="--", lw=1)
    if ratio:
        ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in rows], fontsize=7)
    ax.set_xlabel(f"{rows[0].measure} (emulated o, observed x; null={null})")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
