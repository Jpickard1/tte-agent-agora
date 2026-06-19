"""Per-(nct_id, dataset) CONTEXT record + JSONL persistence (#95, worker1).

The corpus JSONL (contracts/io.py) is one ``ComparisonResult`` per line — the
WHAT (the emulated estimate vs observed). This is its sidecar: one
``TrialDatasetContext`` per line — the WHY behind each gallery row:

  * emulability (#35)   — is it emulable here, score, sepsis flag, reasons;
  * measurability (#33) — per-element measurable/proxy/unmeasurable + gaps;
  * proxy + missingness (#34) — surrogates used + data availability;
  * variability (#32)   — the trial-level cross-dataset heterogeneity +
    attribution (why concordant/divergent), denormalized onto each dataset row.

JOIN KEY: ``(nct_id, dataset)`` — tte1's #49 UI / #40 cards and probe's #61
drivers join this sidecar to the corpus on that pair. Schema: one
``TrialDatasetContext.model_dump_json()`` per line, streaming (O(1) memory).

Import-light (lives in contracts/, only pydantic) so the UI can load context
WITHOUT importing the analysis/triage layer — mirroring contracts/io.py.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class TrialDatasetContext(BaseModel):
    """The WHY for one (trial, dataset) gallery row. Flat headline fields for easy
    filtering/sorting + nested blocks for detail. Joined to the corpus by
    (nct_id, dataset)."""

    nct_id: str
    dataset: str
    is_sepsis: bool = False

    # --- #35 emulability (headline + full score) ---
    emulable: bool = False
    emulability_score: float = 0.0
    emulability: dict = Field(default_factory=dict)        # EmulabilityScore.model_dump()

    # --- #33 protocol-vs-data measurability ---
    measurability: dict = Field(default_factory=dict)      # DatasetMeasurability.summary

    # --- #34 proxy substitution + data missingness ---
    proxy_list: list[dict] = Field(default_factory=list)
    missingness: dict | None = None                        # present only when a frame was supplied

    # --- #32 cross-dataset variability (TRIAL-level; same on each dataset row) ---
    variability: dict | None = None


def dump_context_jsonl(records, path) -> int:
    """Persist a TrialDatasetContext stream to JSONL (one model_dump_json per
    line), alongside the corpus JSONL. Streams; returns the count written."""
    n = 0
    with open(path, "w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def load_context_jsonl(path):
    """Iterate TrialDatasetContext from a JSONL sidecar (lazy). The UI/drivers
    index these by (nct_id, dataset) to join the corpus."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield TrialDatasetContext.model_validate_json(line)


# --------------------------------------------------------------------------- #
# Display layer (#95/#49): a pure, import-light transform of context records into
# a render-ready model — the same "you distill, I render" split as forest_rows /
# the #60 figure data. tte1's #49 cards consume this; NO matplotlib/analysis.
# --------------------------------------------------------------------------- #

def _why_emulable(emu: dict) -> str:
    """One-liner from the #35 emulability block."""
    score = emu.get("score", 0.0)
    if emu.get("emulable"):
        return f"Emulable (score {score:.2f}): eligibility, exposure and outcome reconstructable here."
    reasons = emu.get("reasons") or []
    tail = f" — {reasons[0]}" if reasons else ""
    return f"Not emulable (score {score:.2f}){tail}"


class PanelRow(BaseModel):
    """Render-ready 'why' row for one gallery card, keyed by (nct_id, dataset).
    Optional what-fields (agreement/estimates) are filled when `comparisons` is
    supplied, so a card shows what AND why together."""
    nct_id: str
    dataset: str
    is_sepsis: bool = False
    emulable: bool = False
    emulability_score: float = 0.0
    why_emulable: str = ""
    measurability: dict = Field(default_factory=dict)
    proxy_list: list[dict] = Field(default_factory=list)
    why_divergent: str | None = None              # one-line #32 attribution note
    missingness: dict | None = None
    agreement: str | None = None                  # from optional comparisons join
    emulated_estimate: float | None = None
    observed_estimate: float | None = None


class CorpusRollup(BaseModel):
    """Corpus-level summary panel (one per gallery)."""
    n_rows: int = 0
    n_trials: int = 0
    n_sepsis_rows: int = 0
    pct_fully_measurable: float = 0.0
    mean_emulability: float = 0.0
    count_by_emulable: dict = Field(default_factory=dict)
    top_proxy_elements: list[dict] = Field(default_factory=list)


class ContextPanel(BaseModel):
    rows: list[PanelRow] = Field(default_factory=list)
    rollup: CorpusRollup = Field(default_factory=CorpusRollup)


def context_panel(context_records, *, comparisons=None) -> ContextPanel:
    """Distill TrialDatasetContext records (#95) into a render-ready ContextPanel
    for tte1's #49 cards: a per-(nct_id,dataset) 'why' row + a corpus rollup. Pure
    + import-light. `comparisons` (optional ComparisonResult iterable) joins the
    'what' (agreement + estimates) onto each row by (nct_id, dataset)."""
    recs = list(context_records)
    comp_idx = {}
    for c in (comparisons or []):
        comp_idx[(c.nct_id, c.dataset)] = c

    rows: list[PanelRow] = []
    for r in recs:
        why_div = None
        if isinstance(r.variability, dict):
            why_div = (r.variability.get("attribution") or {}).get("note")
        c = comp_idx.get((r.nct_id, r.dataset))
        agreement = getattr(c.agreement, "value", None) if c is not None else None
        rows.append(PanelRow(
            nct_id=r.nct_id, dataset=r.dataset, is_sepsis=r.is_sepsis,
            emulable=r.emulable, emulability_score=r.emulability_score,
            why_emulable=_why_emulable(r.emulability or {}),
            measurability=r.measurability, proxy_list=r.proxy_list,
            why_divergent=why_div, missingness=r.missingness,
            agreement=agreement,
            emulated_estimate=(c.emulated.estimate if c is not None else None),
            observed_estimate=(c.observed_estimate if c is not None else None),
        ))

    n = len(recs)
    n_fully = sum(1 for r in recs if (r.measurability or {}).get("fully_measurable"))
    proxy_counts: dict[str, int] = {}
    for r in recs:
        for p in r.proxy_list:
            key = p.get("concept", "?")
            proxy_counts[key] = proxy_counts.get(key, 0) + 1
    top = sorted(proxy_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    rollup = CorpusRollup(
        n_rows=n, n_trials=len({r.nct_id for r in recs}),
        n_sepsis_rows=sum(1 for r in recs if r.is_sepsis),
        pct_fully_measurable=round(n_fully / n, 4) if n else 0.0,
        mean_emulability=round(sum(r.emulability_score for r in recs) / n, 4) if n else 0.0,
        count_by_emulable={"emulable": sum(1 for r in recs if r.emulable),
                           "not_emulable": sum(1 for r in recs if not r.emulable)},
        top_proxy_elements=[{"concept": k, "count": v} for k, v in top],
    )
    return ContextPanel(rows=rows, rollup=rollup)


__all__ = ["TrialDatasetContext", "dump_context_jsonl", "load_context_jsonl",
           "PanelRow", "CorpusRollup", "ContextPanel", "context_panel"]
