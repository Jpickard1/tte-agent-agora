"""Protocol-vs-data measurability / gap analysis (#33, worker1).

For each trial, classify EVERY protocol element — eligibility predicate, arm
exposure, outcome, covariate — as **measurable**, **proxy**, or **unmeasurable**
in each dataset, and surface the gaps. This is the "what can we definitively know
vs not" layer: where #35 gives a scalar emulability SCORE per (trial, dataset),
this gives the element-by-element REPORT behind it.

Grounding (truthful, not aspirational): an element's status is driven by whether
its EVENT_TYPE is actually captured by that dataset's adapter (#6/#7/#8):

  * DIRECT  — the adapter emits this event domain today            -> measurable
  * PROXY   — the domain exists in the source DB but the adapter
              doesn't extract it yet (a surrogate/wiring gap)      -> proxy
  * neither — the domain isn't in that dataset at all              -> unmeasurable

with two refinements: a non-EHR/soft concept (quality-of-life, questionnaire…) is
always unmeasurable, and an OUTCOME is judged by whether it's a hard EHR endpoint
(mortality/LOS/…) — reusing #35's keyword set so the two stay consistent.

The DIRECT sets mirror each adapter's TABLE_SPEC (+ MIMIC's deathtime OUTCOME);
``tests/test_measurability.py`` drift-checks them against the adapters when pandas
is present. Pure stdlib — no pandas — so it runs in CI's [dev] env.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts.events import EventType
from .contracts.trial_spec import TargetTrialSpec
from .triage import MEASURABLE_OUTCOME_KEYWORDS

MEASURABLE, PROXY, UNMEASURABLE = "measurable", "proxy", "unmeasurable"

#: event domains each dataset's ADAPTER emits today (mirror of TABLE_SPEC; MIMIC
#: also emits OUTCOME via admissions.deathtime, PR #54).
DATASET_DIRECT: dict[str, set[EventType]] = {
    "MIMIC-IV": {EventType.DIAGNOSIS, EventType.LAB, EventType.MEASUREMENT,
                 EventType.MEDICATION, EventType.OUTCOME},
    # eICU now emits vitals (MEASUREMENT) + mortality (OUTCOME) too — #83 wired
    # vitalPeriodic/Aperiodic + patient discharge status, so they're DIRECT.
    "eICU-CRD": {EventType.DIAGNOSIS, EventType.LAB, EventType.MEDICATION,
                 EventType.MEASUREMENT, EventType.OUTCOME},
    "MGB": {EventType.DIAGNOSIS, EventType.LAB, EventType.MEDICATION},  # gated, conservative
}

#: domains present in the source DB but NOT yet extracted by the adapter — a
#: surrogate is available / it needs wiring (so: proxy, not flatly unmeasurable).
DATASET_PROXY: dict[str, set[EventType]] = {
    "MIMIC-IV": {EventType.DEMOGRAPHIC, EventType.PROCEDURE, EventType.LOCATION},
    "eICU-CRD": {EventType.DEMOGRAPHIC, EventType.PROCEDURE, EventType.LOCATION},
    # MGB (gated): vitals/mortality exist in the source but the adapter is a
    # passthrough filter, not yet wired for them.
    "MGB": {EventType.MEASUREMENT, EventType.OUTCOME, EventType.DEMOGRAPHIC,
            EventType.PROCEDURE},
}

#: concepts that aren't reconstructable from ICU EHR at all (patient-reported /
#: instrument-based) -> unmeasurable regardless of event type.
_SOFT_CONCEPT_KEYWORDS = (
    "quality of life", "qol", "questionnaire", "satisfaction", "well-being",
    "wellbeing", "anxiety scale", "depression scale", "adherence", "self-report",
    "patient-reported", "eq-5d", "whoqol", "sf-36",
)


@dataclass
class ElementMeasurability:
    """One protocol element's status in one dataset."""
    kind: str            # eligibility | exposure | outcome | covariate
    concept: str
    event_type: str
    status: str          # measurable | proxy | unmeasurable
    reason: str


@dataclass
class DatasetMeasurability:
    """Per-(trial, dataset) measurability report."""
    nct_id: str
    dataset: str
    elements: list[ElementMeasurability] = field(default_factory=list)

    def _count(self, status: str) -> int:
        return sum(1 for e in self.elements if e.status == status)

    @property
    def summary(self) -> dict:
        n = len(self.elements)
        n_meas = self._count(MEASURABLE)
        return {
            "nct_id": self.nct_id,
            "dataset": self.dataset,
            "n_elements": n,
            "n_measurable": n_meas,
            "n_proxy": self._count(PROXY),
            "n_unmeasurable": self._count(UNMEASURABLE),
            "fully_measurable": n > 0 and n_meas == n,
            "measurable_fraction": 0.0 if n == 0 else round(n_meas / n, 4),
            # the gaps, surfaced (never hidden): the non-measurable elements
            "gaps": [{"kind": e.kind, "concept": e.concept, "status": e.status,
                      "reason": e.reason} for e in self.elements if e.status != MEASURABLE],
        }


def _is_soft(concept: str) -> bool:
    c = (concept or "").lower()
    return any(k in c for k in _SOFT_CONCEPT_KEYWORDS)


def _classify(concept: str, event_type: EventType, dataset: str, *, is_outcome: bool = False) -> tuple[str, str]:
    """(status, reason) for one element in one dataset."""
    direct = DATASET_DIRECT.get(dataset, set())
    proxy = DATASET_PROXY.get(dataset, set())

    if _is_soft(concept):
        return UNMEASURABLE, f"'{concept}' is patient-reported / instrument-based — not in ICU EHR"

    if is_outcome:
        c = (concept or "").lower()
        if not any(k in c for k in MEASURABLE_OUTCOME_KEYWORDS):
            return UNMEASURABLE, f"outcome '{concept}' is not a hard EHR endpoint (no mortality/LOS/… signal)"
        # a hard endpoint: measurable iff the dataset emits OUTCOME, else proxy
        if EventType.OUTCOME in direct:
            return MEASURABLE, f"hard endpoint; {dataset} adapter emits OUTCOME"
        if EventType.OUTCOME in proxy:
            return PROXY, (f"hard endpoint, but {dataset} adapter doesn't extract OUTCOME yet "
                           "(available via discharge status — needs wiring)")
        return UNMEASURABLE, f"{dataset} has no OUTCOME source"

    if event_type in direct:
        return MEASURABLE, f"{dataset} adapter captures {event_type.value} events"
    if event_type in proxy:
        return PROXY, (f"{event_type.value} exists in {dataset} but the adapter doesn't extract it "
                       "yet (surrogate / needs wiring)")
    return UNMEASURABLE, f"{dataset} captures no {event_type.value} events"


def measurability_report(spec: TargetTrialSpec, dataset: str) -> DatasetMeasurability:
    """Classify every protocol element of `spec` against `dataset`."""
    rep = DatasetMeasurability(nct_id=spec.nct_id, dataset=dataset)

    def add(kind: str, concept: str, et: EventType, *, is_outcome: bool = False) -> None:
        status, reason = _classify(concept, et, dataset, is_outcome=is_outcome)
        rep.elements.append(ElementMeasurability(kind=kind, concept=concept,
                                                 event_type=et.value, status=status, reason=reason))

    for c in spec.eligibility:
        add("eligibility", c.concept, c.event_type)
    for arm in spec.arms:
        for iv in arm.intervention_concepts:
            add("exposure", iv, EventType.MEDICATION)
    for o in spec.outcomes:
        add("outcome", o.concept or o.name, o.event_type, is_outcome=True)
    for cov in spec.covariates:
        add("covariate", cov, EventType.MEASUREMENT)  # covariates are bare names; vitals/labs domain
    return rep


def build_measurability_catalog(
    specs: list[TargetTrialSpec], *, datasets: tuple[str, ...] = ("MIMIC-IV", "eICU-CRD"),
) -> dict:
    """A measurability report per (trial, dataset) + an aggregate summary. Acceptance
    (#33): the per-trial-per-DB report. Surfaces the gap distribution, never hides it."""
    reports: list[DatasetMeasurability] = []
    for spec in specs:
        for ds in datasets:
            reports.append(measurability_report(spec, ds))

    gap_reasons: dict[str, int] = {}
    for r in reports:
        for g in r.summary["gaps"]:
            gap_reasons[g["reason"]] = gap_reasons.get(g["reason"], 0) + 1

    return {
        "reports": [r.summary for r in reports],
        "elements": [
            {"nct_id": r.nct_id, "dataset": r.dataset, "kind": e.kind, "concept": e.concept,
             "event_type": e.event_type, "status": e.status, "reason": e.reason}
            for r in reports for e in r.elements
        ],
        "summary": {
            "n_trials": len(specs),
            "n_reports": len(reports),
            "n_fully_measurable": sum(1 for r in reports if r.summary["fully_measurable"]),
            "gap_reasons": dict(sorted(gap_reasons.items(), key=lambda kv: -kv[1])),
            "datasets": list(datasets),
        },
    }


__all__ = [
    "ElementMeasurability", "DatasetMeasurability", "measurability_report",
    "build_measurability_catalog", "DATASET_DIRECT", "DATASET_PROXY",
    "MEASURABLE", "PROXY", "UNMEASURABLE",
]
