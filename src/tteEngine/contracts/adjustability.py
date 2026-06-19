"""Confounder adjustability ledger + PS diagnostics (#105) — the seam.

Per (nct_id, dataset), makes EXPLICIT and visible which confounders are
ADJUSTED, MEASURABLE-but-not-used, or NOT-ADJUSTABLE (unmeasurable/proxy here ->
residual confounding), and carries the propensity-score overlap / common-support
diagnostic alongside the SMD balance the engine already computes.

Import-light (pydantic only), next to ComparisonResult/TrialDatasetContext, so the
UI (#104 per-trial detail) and the meta-analysis read it WITHOUT the analysis
extra. Persisted as a JSONL sidecar joined to the corpus on (nct_id, dataset) —
the same key as corpus.jsonl / context.jsonl.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Adjustability(str, Enum):
    ADJUSTED = "adjusted"                      # measurable here AND in the PS/outcome model
    MEASURABLE_NOT_USED = "measurable_not_used"  # measurable but not entered into the model
    NOT_ADJUSTABLE = "not_adjustable"         # unmeasurable / proxy-only here -> residual confounding


class ConfounderRow(BaseModel):
    confounder: str
    event_type: str
    status: str = Field(..., description="measurable | proxy | unmeasurable (from #33)")
    classification: Adjustability
    in_model: bool = False
    smd_before: float | None = None
    smd_after: float | None = None
    reason: str = ""


class ConfounderLedger(BaseModel):
    nct_id: str
    dataset: str
    adjustment: str = ""                       # iptw | psm | covariate | unadjusted
    considered: list[ConfounderRow] = Field(default_factory=list)
    n_considered: int = 0
    n_adjusted: int = 0
    n_measurable_not_used: int = 0
    n_not_adjustable: int = 0
    ps_overlap: dict | None = Field(
        None, description="engine ps_overlap: frac_treated_off_support / overlap_coef / poor (+ histogram)."
    )
    e_value_point: float | None = None
    residual_confounding_note: str = ""
    summary_line: str = ""                      # e.g. 'adjusted 6/8 confounders; 2 not-adjustable -> residual (E-value 1.4)'

    def not_adjustable(self) -> list[ConfounderRow]:
        return [r for r in self.considered if r.classification == Adjustability.NOT_ADJUSTABLE]


def dump_ledger_jsonl(ledgers, path) -> int:
    """Persist ConfounderLedger rows to JSONL (one model_dump_json per line)."""
    n = 0
    with open(path, "w") as f:
        for led in ledgers:
            f.write(led.model_dump_json() + "\n")
            n += 1
    return n


def load_ledger_jsonl(path):
    """Stream ConfounderLedger rows back from JSONL (import-light reader)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield ConfounderLedger.model_validate_json(line)
