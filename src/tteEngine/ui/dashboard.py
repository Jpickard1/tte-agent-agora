"""Results dashboard data model (#49) — the headline view, computed once and
rendered by the Streamlit app (or any front-end) WITHOUT matplotlib.

`build_dashboard(comparisons)` assembles everything the gallery shows from the
persisted corpus + the analysis outputs:
  * headline concordance rate + Wilson CI (#64 concordance_summary)
  * pooled effect + I2/tau2 heterogeneity, and the sepsis subgroup (#64 meta_analyze)
  * calibration slope/coverage + scatter points (#41 corpus_calibration)
  * per-trial forest rows (#60 forest_rows) and Trial Emulation Cards (#40)

Analysis is imported LAZILY so importing the ui package stays cheap; the heavy
work runs only when a dashboard is built. Streams the corpus, so it scales to the
live >1k/>10k run (the front-end filters to sepsis by rebuilding on the subset).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel, Field

from tteEngine.figures.forest import ForestRow, forest_rows
from tteEngine.ui.cards import TrialEmulationCard, build_cards

if TYPE_CHECKING:
    from tteEngine.contracts.results import ComparisonResult


class DashboardModel(BaseModel):
    n_total: int = 0
    n_sepsis: int = 0
    concordance: dict = Field(default_factory=dict)        # rate, ci_low, ci_high, n_comparable, n
    pooled: dict = Field(default_factory=dict)             # estimate, ci_low, ci_high, i2, tau2
    sepsis_pooled: dict | None = None
    calibration: dict = Field(default_factory=dict)        # slope, intercept, coverage, pearson_r, rmse, points
    forest_rows: list[ForestRow] = Field(default_factory=list)
    cards: list[TrialEmulationCard] = Field(default_factory=list)
    context_summary: dict = Field(default_factory=dict)    # #98 corpus measurability/emulability rollup


def build_dashboard(
    comparisons: Iterable["ComparisonResult"],
    *,
    sepsis_ncts: set[str] | None = None,
    context_records=None,
    ledger_records=None,
    audit_records=None,
) -> DashboardModel:
    """Assemble the dashboard model from a comparison stream + the analysis outputs.

    `context_records` (optional, #98): worker1's TrialDatasetContext sidecar. When
    given, each card gets a `.why` (joined by (nct_id,dataset)) and the model gets a
    corpus `context_summary`. Import-light join — no analysis dependency.
    """
    from tteEngine.analysis.meta import concordance_summary, meta_analyze
    from tteEngine.analysis.reliability import corpus_calibration

    rows = list(comparisons)  # iterated by several aggregators
    sepsis = sepsis_ncts or set()

    conc = concordance_summary(rows)
    meta = meta_analyze(rows, subgroup=(lambda c: "sepsis" if c.nct_id in sepsis else "other")
                        if sepsis else None)
    calib = corpus_calibration(rows)

    sepsis_pooled = None
    for sg in meta.by_subgroup:
        if getattr(sg, "name", None) == "sepsis":
            pe = sg.pooled_effect
            sepsis_pooled = {"estimate": pe.pooled_estimate, "ci_low": pe.ci_low,
                             "ci_high": pe.ci_high, "i2": pe.i2, "k": pe.k}

    ledger_index = None
    if ledger_records is not None:
        def _k(rec):
            return (rec.get("nct_id"), rec.get("dataset")) if isinstance(rec, dict) \
                else (getattr(rec, "nct_id", None), getattr(rec, "dataset", None))
        ledger_index = {_k(r): r for r in ledger_records}
    cards = build_cards(rows, sepsis_ncts=sepsis, ledger_index=ledger_index)
    context_summary: dict = {}
    if context_records is not None:
        from tteEngine.ui.context_panel import corpus_context_summary, index_context, why_for
        ctx_recs = list(context_records)
        idx = index_context(ctx_recs)
        for card in cards:
            ctx = idx.get((card.nct_id, card.dataset))
            if ctx is not None:
                card.why = why_for(ctx)
        context_summary = corpus_context_summary(ctx_recs)

    if audit_records is not None:  # #130: join the assignment audit by (nct_id, dataset)
        def _ak(rec):
            return (rec.get("nct_id"), rec.get("dataset")) if isinstance(rec, dict) \
                else (getattr(rec, "nct_id", None), getattr(rec, "dataset", None))
        audit_index = {_ak(r): (r if isinstance(r, dict) else r.model_dump()) for r in audit_records}
        for card in cards:
            card.audit = audit_index.get((card.nct_id, card.dataset))

    from tteEngine.figures.calibration import calibration_points
    return DashboardModel(
        n_total=len(rows),
        n_sepsis=sum(1 for c in rows if c.nct_id in sepsis),
        concordance={"rate": conc.rate, "ci_low": conc.ci_low, "ci_high": conc.ci_high,
                     "n_comparable": conc.n_comparable, "n": conc.n,
                     "n_concordant": conc.n_concordant},
        pooled={"estimate": meta.pooled_effect.pooled_estimate, "ci_low": meta.pooled_effect.ci_low,
                "ci_high": meta.pooled_effect.ci_high, "i2": meta.pooled_effect.i2,
                "tau2": meta.pooled_effect.tau2, "k": meta.pooled_effect.k},
        sepsis_pooled=sepsis_pooled,
        calibration={"slope": calib.slope, "intercept": calib.intercept,
                     "coverage": calib.coverage, "pearson_r": calib.pearson_r,
                     "rmse": calib.rmse, "n": calib.n, "points": calibration_points(calib)},
        forest_rows=forest_rows(rows),
        cards=cards,
        context_summary=context_summary,
    )


def ctgov_url(nct_id: str) -> str:
    """Real ClinicalTrials.gov study link (opens the registered trial)."""
    return f"https://clinicaltrials.gov/study/{nct_id}"


def trial_table(model: "DashboardModel") -> list[dict]:
    """Flat one-row-per-(trial,dataset) table for the sortable/filterable summary
    view. Carries the ctgov link + the WHY headline (emulable/score) so the table
    cross-links to ClinicalTrials.gov and to the per-trial detail."""
    rows = []
    for c in model.cards:
        why = c.why or {}
        conf = (c.confounders or {}).get("summary") or {}
        rows.append({
            "nct_id": c.nct_id,
            "ctgov": ctgov_url(c.nct_id),
            "dataset": c.dataset,
            "sepsis": c.is_sepsis,
            "measure": c.measure,
            "emulated": c.emulated_estimate,
            "ci_low": c.ci_low,
            "ci_high": c.ci_high,
            "observed": c.observed_estimate,
            "agreement": c.agreement,
            "emulable": why.get("emulable"),
            "score": why.get("emulability_score"),
            "adjusted": conf.get("label"),  # e.g. "adjusted 6/8" (#104 confounder summary)
        })
    return rows


def group_by_trial(model: "DashboardModel") -> dict[str, list[TrialEmulationCard]]:
    """Group cards by NCT id (preserving order) for the per-trial detail view —
    one trial may appear across multiple datasets."""
    out: dict[str, list[TrialEmulationCard]] = {}
    for c in model.cards:
        out.setdefault(c.nct_id, []).append(c)
    return out
