"""Batch TTE over the emulable-trial corpus (#36, tte1).

Runs the full spine — study -> spec -> ExtractionPlan -> per-DB extract -> cohort
-> TTE -> emulated-vs-observed — over every trial x dataset, and STREAMS the
ComparisonResult rows into the #11 benchmark aggregator. This is the engine that
produces the >1k/>10k-TTE gallery.

Design guarantees (per jpic's directive):
  * STREAMING: `run_corpus` is a generator; `run_corpus_benchmark` feeds it
    straight into `run_benchmark`, so memory is O(datasets) regardless of corpus
    size (millions of rows -> only counters retained).
  * SEPSIS-FIRST: ordering comes from `ctgov.fetch_corpus(sepsis_first=True)`;
    the runner preserves input order, so sepsis trials are processed/ranked first.
  * NO SILENT CAPS: every trial x dataset that can't produce a row is recorded in
    a `DropLog` with an explicit reason (never dropped silently). Any explicit cap
    (max_studies) lives in the caller's fetch_corpus and is visible.

Import-light: ctgov + analysis are imported lazily, so importing this module
(and orchestration) does not pull the heavy `analysis` extra. Per-DB extraction
and the TTE estimator are INJECTED (synthetic for offline/CI; real adapters +
engine for the live corpus run), mirroring the Pipeline provider pattern.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

from tteEngine.cohort import build_cohort
from tteEngine.contracts.results import ComparisonResult

if TYPE_CHECKING:
    from tteEngine.contracts.trial_spec import TargetTrialSpec


class DropLog:
    """Records every (trial, dataset) that did not yield a comparison row, with a
    reason. The anti-silent-cap ledger: what was NOT in the gallery and why."""

    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, nct_id: str, dataset: str, reason: str) -> None:
        self.items.append({"nct_id": nct_id, "dataset": dataset, "reason": reason})

    def by_reason(self) -> dict[str, int]:
        c: Counter = Counter(d["reason"] for d in self.items)
        return dict(c)

    def __len__(self) -> int:
        return len(self.items)


def _arms_ok(cohort) -> bool:
    treated = sum(len(a.trajectory_ids) for a in cohort.arms if not a.is_control)
    control = sum(len(a.trajectory_ids) for a in cohort.arms if a.is_control)
    return treated > 0 and control > 0


def assignment_audit_from_cohort(cohort):
    """Assemble a contracts.audit.AssignmentAudit from a CohortResult's #138 audit
    primitives — the 'how patients were sorted' record live_run persists (audit.jsonl)
    and the #130 UI renders. Pure mapping; no recomputation of include/drop."""
    from tteEngine.contracts.audit import (ArmAudit, AssignmentAudit, EligibilityDecision,
                                           MatchProvenance)
    d = cohort.diagnostics
    arms = [ArmAudit(name=a.name, is_control=a.is_control, n=len(a.trajectory_ids),
                     match_method_counts=cohort.arm_method_counts.get(a.name, {}))
            for a in cohort.arms]
    return AssignmentAudit(
        nct_id=cohort.nct_id, dataset=cohort.dataset,
        n_screened=getattr(d, "n_screened", 0), n_eligible=getattr(d, "n_eligible", 0),
        n_enrolled=cohort.n_total, n_excluded_immortal=getattr(d, "n_excluded_immortal", 0),
        n_unassigned=cohort.n_unassigned, arms=arms,
        eligibility=[EligibilityDecision(**e) for e in cohort.eligibility_decisions],
        sample=[MatchProvenance(**p) for p in cohort.assignment_provenance],
        n_low_confidence=cohort.n_low_confidence,
    )


def run_corpus(
    jobs: Iterable[tuple[dict, "TargetTrialSpec"]],
    datasets: list[str],
    *,
    extract_fn: Callable,
    engine_fn: Callable,
    compare_fn: Callable | None = None,
    plan_fn: Callable | None = None,
    drops: DropLog | None = None,
    resolve=None,
    arm_match_fn=None,
    measurable_fn=None,
    on_audit: Callable | None = None,
) -> Iterator[ComparisonResult]:
    """Stream one ComparisonResult per (trial x dataset) that emulates cleanly.

    jobs: iterable of (study_dict, TargetTrialSpec) — e.g. from
      ``[(s, study_to_spec(s)) for s in fetch_corpus(sepsis_first=True)]``.
    extract_fn(plan, spec, dataset) -> canonical 5-col DataFrame | None
      (inject a real adapter for live data; a synthetic generator offline).
    engine_fn(events, cohort, spec) -> contracts.TTEResult
      (e.g. make_engine_provider([...]) for IPTW, or the crude baseline).
    Drops (no extractable events / empty cohort / missing arm / engine error) are
    recorded in `drops`, never silently skipped.
    """
    from tteEngine.analysis import compare_trial
    from tteEngine.ctgov import spec_to_plan

    compare_fn = compare_fn or compare_trial
    plan_fn = plan_fn or spec_to_plan
    drops = drops if drops is not None else DropLog()

    for study, spec in jobs:
        nct = spec.nct_id
        for ds in datasets:
            try:
                plan = plan_fn(spec, dataset=ds)
                events = extract_fn(plan, spec, ds)
                if events is None or len(events) == 0:
                    drops.add(nct, ds, "no extractable events")
                    continue
                cohort = build_cohort(events, spec, dataset=ds, resolve=resolve,
                                      arm_match_fn=arm_match_fn, measurable_fn=measurable_fn)
                if cohort.n_total == 0:
                    drops.add(nct, ds, "empty cohort after eligibility")
                    continue
                if not _arms_ok(cohort):
                    drops.add(nct, ds, "missing treated or control arm")
                    continue
                if on_audit is not None:  # #138/#130: surface the AssignmentAudit for live_run to persist
                    on_audit(assignment_audit_from_cohort(cohort))
                emulated = engine_fn(events, cohort, spec)
                yield compare_fn(study, emulated, dataset=ds)
            except Exception as exc:  # one trial must never kill the batch
                drops.add(nct, ds, f"error: {type(exc).__name__}: {exc}")
                continue


def run_corpus_benchmark(
    jobs: Iterable[tuple[dict, "TargetTrialSpec"]],
    datasets: list[str],
    **kwargs,
) -> tuple[dict, DropLog]:
    """Stream the corpus through the #11 benchmark; return (summary, drops).

    Memory is O(datasets): only counters + the drop ledger are retained, so this
    scales to the full >1k/>10k corpus. The summary carries the dropped-count +
    drops-by-reason so coverage is never silently overstated.
    """
    from tteEngine.analysis import run_benchmark

    drops = kwargs.pop("drops", None) or DropLog()
    summary = run_benchmark(run_corpus(jobs, datasets, drops=drops, **kwargs))
    summary["n_dropped"] = len(drops)
    summary["drops_by_reason"] = drops.by_reason()
    return summary, drops


def run_corpus_to_jsonl(jobs, datasets, path, *, drops: DropLog | None = None, **kwargs):
    """Run the corpus and PERSIST every ComparisonResult to JSONL in one streaming
    pass, via the import-light contracts I/O (orchestration never imports analysis).
    Returns (n_written, drops). Read back / analyze OFFLINE from the saved corpus:

        from tteEngine.contracts.io import load_comparisons_jsonl
        from tteEngine.analysis import run_benchmark, meta_analyze
        run_benchmark(load_comparisons_jsonl(path))   # summary
        meta_analyze(load_comparisons_jsonl(path))     # #64 headline + forest
        # figures #60 / UI #49 read the same load_comparisons_jsonl(path)

    This is the bridge from the gated live MIMIC/eICU run to every downstream
    deliverable: the corpus is persisted ONCE and read many times.
    """
    from tteEngine.contracts.io import dump_comparisons_jsonl

    drops = drops if drops is not None else DropLog()
    n = dump_comparisons_jsonl(run_corpus(jobs, datasets, drops=drops, **kwargs), path)
    return n, drops
