"""Assignment-audit schema (#139) — the seam that makes patient grouping fully
auditable: HOW each patient was sorted into the cohort + arms.

This is the contract worker1's code-based matcher (#136/#137) + probe's cohort
integration emit into, and the #130 UI renders. Import-light (no analysis/pandas),
joined to the corpus by (nct_id, dataset) like context.jsonl / ledger.jsonl, and
persisted as a sidecar (audit.jsonl) by the live run / reproduce.

Confidence tiers (LOCKED, in descending trust):
    rxnorm_code  > ingredient > name > substring
String/substring is the last-resort tier — surfaced AMBER in the UI, never buried.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    """How a real event was matched to a trial concept (drug/dx/lab)."""

    RXNORM_CODE = "rxnorm_code"     # drug matched on a resolved RxNorm code (highest trust)
    ICD_HIERARCHY = "icd_hierarchy"  # dx matched on ICD code family/hierarchy (structured-code, top trust)
    INGREDIENT = "ingredient"       # matched at ingredient level (brand/salt/route rolled up)
    NAME = "name"                   # exact normalized name match
    SUBSTRING = "substring"         # last-resort string/substring match (LOW — render amber)


#: tiers the UI should flag as low-confidence.
LOW_CONFIDENCE = {Confidence.SUBSTRING}


class MatchProvenance(BaseModel):
    """Why ONE patient landed in an arm: the real event + the code/method that
    matched it to the trial intervention. Emitted by worker1's matcher (#136)."""

    trajectory_id: int
    arm: str
    matched_event_name: str | None = None     # the raw EHR event name (e.g. 'hydrocortisone na succ.')
    matched_code: str | None = None           # the code that matched (e.g. RxNorm 5492, GSN, NDC)
    concept: str | None = None                # the trial intervention it matched ('Drug: Hydrocortisone')
    method: Confidence = Confidence.SUBSTRING
    t_rel_hours: float | None = None          # time of the matched event relative to t0
    source_table: str | None = None           # which EHR table the event came from (worker1: prescriptions/diagnoses_icd/...)
    matched_row_id: str | None = None          # the specific event row (probe fills at assembly)
    dose: str | None = None                    # event-level (probe fills): e.g. '100 mg'
    route: str | None = None                   # event-level (probe fills): e.g. 'IV'


class EligibilityDecision(BaseModel):
    """Per-criterion eligibility outcome — enforced/met/failed vs honestly skipped."""

    concept: str | None = None
    event_type: str | None = None
    comparator: str | None = None
    value: float | str | list | None = None
    measurable: bool = True                   # is this criterion assessable in this dataset?
    result: str = "applied"                   # "met" | "failed" | "skipped_unmeasurable"
    reason: str | None = None                 # why skipped, or how resolved


class ArmAudit(BaseModel):
    """Per-arm assignment summary: which codes defined it + how patients matched."""

    name: str
    is_control: bool = False
    n: int = 0
    defining_codes: list[str] = Field(default_factory=list)   # the resolved code-set defining this arm
    intervention_concepts: list[str] = Field(default_factory=list)
    match_method_counts: dict[str, int] = Field(default_factory=dict)  # Confidence.value -> count


class AssignmentAudit(BaseModel):
    """The full 'HOW PATIENTS WERE SORTED' record for one (trial, dataset).

    Reconciles: n_screened = n_excluded_ineligible + n_excluded_immortal + n_unassigned + sum(arm.n).
    Emitted per (nct_id, dataset); rendered by the #130 UI; persisted in audit.jsonl.
    """

    nct_id: str
    dataset: str
    n_screened: int = 0
    n_eligible: int = 0
    n_enrolled: int = 0
    n_excluded_immortal: int = 0
    n_unassigned: int = 0                      # eligible but matched no arm
    arms: list[ArmAudit] = Field(default_factory=list)
    eligibility: list[EligibilityDecision] = Field(default_factory=list)
    sample: list[MatchProvenance] = Field(default_factory=list)  # bounded per-patient examples
    n_low_confidence: int = 0                  # arm matches at a LOW_CONFIDENCE tier (UI ambers these)

    def match_method_totals(self) -> dict[str, int]:
        """Corpus-renderable rollup of match methods across arms (e.g. '38 rxnorm_code · 2 substring')."""
        totals: dict[str, int] = {}
        for arm in self.arms:
            for method, n in arm.match_method_counts.items():
                totals[method] = totals.get(method, 0) + n
        return totals


def dump_audit_jsonl(records, path) -> int:
    """Persist an AssignmentAudit stream to JSONL (one model_dump_json per line),
    alongside corpus.jsonl. Streams; returns the count written."""
    n = 0
    with open(path, "w") as fh:
        for r in records:
            fh.write(r.model_dump_json())
            fh.write("\n")
            n += 1
    return n


def load_audit_jsonl(path):
    """Stream AssignmentAudit back from a JSONL sidecar (lazy; blank-line tolerant)."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield AssignmentAudit.model_validate_json(line)
