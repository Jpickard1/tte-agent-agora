"""TargetTrialSpec -> ExtractionPlan: the extraction-intelligence (#3, probe).

THE generalizer at the heart of jpic's vision: instead of hand-coding one
extraction script per trial per database (trialsim's simulations/2024/<NCT>/
*_mimic_code.py / *_eicu_code.py), this turns a parsed TargetTrialSpec (#2) into
a single, dataset-AGNOSTIC ExtractionPlan — the set of concepts/events each
per-DB adapter (#6/#7/#8) must pull, with their roles, value requirements, and
the cohort-defining filter. Adapters resolve the plan's concepts to their own
schema via the vocab layer (#5); here we stay dataset-agnostic.

Self-contained: this is new code (the patterns are learned from trialsim's
per-trial scripts, not imported). The vocab resolver is pluggable so this works
today with raw concept names and gains cross-DB code resolution once #5 lands.
"""
from __future__ import annotations

from typing import Callable

from ..contracts import (
    Comparator,
    ConceptRequest,
    EventType,
    ExtractionPlan,
    TargetTrialSpec,
)

# A vocab resolver: (raw_concept, event_type) -> normalized concept id. The
# default is identity (raw name passes through); worker1's #5 layer plugs in here.
VocabResolver = Callable[[str, EventType], str]

# comparators that imply a numeric threshold (so the adapter must carry value+unit)
_VALUE_COMPARATORS = {Comparator.GT, Comparator.GE, Comparator.LT, Comparator.LE}

# baseline covariates always worth adjusting for if present in the data
_DEFAULT_COVARIATES: tuple[tuple[str, EventType], ...] = (
    ("age", EventType.DEMOGRAPHIC),
    ("sex", EventType.DEMOGRAPHIC),
)


def _identity_vocab(concept: str, event_type: EventType) -> str:
    return concept


def spec_to_plan(
    spec: TargetTrialSpec,
    *,
    vocab: VocabResolver | None = None,
    dataset: str | None = None,
) -> ExtractionPlan:
    """Build a (dataset-agnostic by default) ExtractionPlan from a TargetTrialSpec.

    Roles: eligibility predicates -> eligibility; arm interventions -> exposure;
    outcomes -> outcome; spec.covariates + defaults -> covariate. `require_value`
    is set when a numeric threshold is involved (lab/measurement eligibility).
    """
    resolve = vocab or _identity_vocab
    concepts: list[ConceptRequest] = []
    seen: set[tuple[str, str]] = set()  # (concept, role) dedup

    def add(raw: str, et: EventType, role: str, require_value: bool) -> None:
        cid = resolve(raw, et)
        key = (cid, role)
        if not cid or key in seen:
            return
        seen.add(key)
        concepts.append(
            ConceptRequest(concept=cid, event_type=et, role=role, require_value=require_value)
        )

    # eligibility predicates
    for c in spec.eligibility:
        needs_value = (c.comparator in _VALUE_COMPARATORS) or (c.value is not None)
        add(c.concept, c.event_type, "eligibility", needs_value)

    # treatment / comparator exposures
    for arm in spec.arms:
        for iv in arm.intervention_concepts:
            add(iv, EventType.MEDICATION, "exposure", require_value=False)

    # outcomes (value needed for continuous outcomes)
    for o in spec.outcomes:
        add(o.concept or o.name, o.event_type, "outcome",
            require_value=(o.kind == "continuous"))

    # covariates: explicit + defaults
    for cov in spec.covariates:
        add(cov, EventType.MEASUREMENT, "covariate", require_value=True)
    for raw, et in _DEFAULT_COVARIATES:
        add(raw, et, "covariate", require_value=False)

    # the cohort-defining concepts (condition + inclusion concepts)
    cohort_filter: list[str] = []
    if spec.condition:
        cohort_filter.append(resolve(spec.condition, EventType.DIAGNOSIS))
    for c in spec.eligibility:
        if c.include and c.event_type in (EventType.DIAGNOSIS, EventType.MEDICATION):
            cohort_filter.append(resolve(c.concept, c.event_type))

    # extraction window from the time-zero grace window (landmark-safe downstream)
    grace = spec.time_zero.grace_window_hours
    window = (-48.0, max(24.0, grace))

    return ExtractionPlan(
        nct_id=spec.nct_id,
        dataset=dataset,
        concepts=concepts,
        cohort_filter_concepts=list(dict.fromkeys(c for c in cohort_filter if c)),
        window_hours=window,
        notes=f"Auto-generated from TargetTrialSpec for {spec.nct_id}"
        + (f" (dataset={dataset})" if dataset else " (dataset-agnostic)"),
    )
