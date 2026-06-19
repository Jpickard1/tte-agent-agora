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


__all__ = ["TrialDatasetContext", "dump_context_jsonl", "load_context_jsonl"]
