"""The resumable per-trial TTE pipeline (#12, tte1).

Generalizes emulaTTE's pipeline.py: a fixed chain of typed steps

    spec -> plan -> extract -> cohort -> tte -> report

where every step is a thin wrapper over an injectable PROVIDER with a graceful
fallback. Real lane implementations (ctgov #2/#3, adapters #6/#7/#8, engine #10,
benchmark #11) plug in by overriding a provider; with no overrides the pipeline
still runs end-to-end on synthetic data (real cohort builder #9 + a crude-RR
baseline engine), so it is testable before the other lanes land.

Check-&-correct: run_until() a step, inspect/edit its output, resume() — editing
a step selectively invalidates everything downstream and only those re-run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, Field

from tteEngine.cohort import build_analysis_frame, build_cohort
from tteEngine.contracts.cohort import CohortResult
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan
from tteEngine.contracts.results import (
    Agreement,
    ComparisonResult,
    EffectMeasure,
    TTEResult,
)
from tteEngine.contracts.trial_spec import TargetTrialSpec

if TYPE_CHECKING:
    import pandas as pd

STEPS: tuple[str, ...] = ("spec", "plan", "extract", "cohort", "tte", "report")


class TargetRequest(BaseModel):
    """Input to a run. Seed fields enable offline/fallback + test runs."""

    nct_id: str
    dataset: str = "MIMIC-IV"
    seed_spec: TargetTrialSpec | None = None
    observed_estimate: float | None = None
    observed_measure: EffectMeasure | None = None


# --------------------------------------------------------------------------- #
# default providers (fallbacks). Override any of these with a real impl.
# --------------------------------------------------------------------------- #

def _default_spec(request: TargetRequest) -> TargetTrialSpec:
    return request.seed_spec or TargetTrialSpec(nct_id=request.nct_id)


def _default_plan(spec: TargetTrialSpec, dataset: str) -> ExtractionPlan:
    concepts: list[ConceptRequest] = []
    for c in spec.eligibility:
        concepts.append(ConceptRequest(concept=c.concept or c.event_type.value,
                                       event_type=c.event_type, role="eligibility"))
    for o in spec.outcomes:
        concepts.append(ConceptRequest(concept=o.concept or o.name, event_type=o.event_type, role="outcome"))
    return ExtractionPlan(nct_id=spec.nct_id, dataset=dataset, concepts=concepts)


def _default_extract(plan: ExtractionPlan, request: "TargetRequest", seed_events):
    if seed_events is None:
        raise RuntimeError(
            "no extract provider and no seed_events: inject a per-DB adapter (#6/#7/#8) "
            "or pass seed_events to run a fallback/offline pipeline."
        )
    return seed_events


def _default_cohort(events: "pd.DataFrame", spec: TargetTrialSpec, dataset: str) -> CohortResult:
    return build_cohort(events, spec, dataset=dataset)


def _crude_rr_engine(events: "pd.DataFrame", cohort: CohortResult, spec: TargetTrialSpec) -> TTEResult:
    """Baseline engine: crude risk ratio of the first outcome, treated vs control.
    A real estimator (#10, PSM/IPTW/Cox) overrides this provider.
    """
    treated = [a for a in cohort.arms if not a.is_control]
    control = [a for a in cohort.arms if a.is_control]
    n_t = sum(len(a.trajectory_ids) for a in treated)
    n_c = sum(len(a.trajectory_ids) for a in control)
    if not spec.outcomes or n_t == 0 or n_c == 0:
        return TTEResult(nct_id=spec.nct_id, dataset=cohort.dataset, method="crude_rr",
                         measure=EffectMeasure.RR, estimate=float("nan"),
                         n_treated=n_t, n_control=n_c, extra={"note": "insufficient arms/outcomes"})
    frame = build_analysis_frame(events, cohort, spec)
    outcome = spec.outcomes[0]
    col = f"outcome_{outcome.name.replace(' ', '_')}"
    treated_ids = {tid for a in treated for tid in a.trajectory_ids}
    f = frame.set_index("TRAJECTORY_ID")
    rt = f.loc[list(treated_ids), col].astype(float).mean()
    rc = f.loc[[t for t in f.index if t not in treated_ids], col].astype(float).mean()
    rr = float("inf") if rc == 0 else float(rt / rc)
    return TTEResult(nct_id=spec.nct_id, dataset=cohort.dataset, method="crude_rr",
                     measure=EffectMeasure.RR, estimate=rr, n_treated=n_t, n_control=n_c,
                     extra={"risk_treated": float(rt), "risk_control": float(rc)})


def _default_report(tte: TTEResult, request: "TargetRequest") -> ComparisonResult:
    obs = request.observed_estimate
    agreement = Agreement.INCONCLUSIVE
    notes = None
    if obs is not None and tte.estimate == tte.estimate:  # not NaN
        # concordant if same side of the null (1.0 for ratios) within a loose band
        null = 1.0 if tte.measure in (EffectMeasure.RR, EffectMeasure.OR, EffectMeasure.HR) else 0.0
        same_side = (tte.estimate - null) * (obs - null) > 0
        agreement = Agreement.CONCORDANT if same_side else Agreement.DISCORDANT
        notes = f"emulated {tte.estimate:.3f} vs observed {obs:.3f} ({tte.measure.value})"
    return ComparisonResult(nct_id=tte.nct_id, dataset=tte.dataset, emulated=tte,
                            observed_estimate=obs, observed_measure=request.observed_measure,
                            agreement=agreement, notes=notes)


def default_providers() -> dict[str, Callable]:
    return {
        "spec": _default_spec,
        "plan": _default_plan,
        "extract": _default_extract,
        "cohort": _default_cohort,
        "tte": _crude_rr_engine,
        "report": _default_report,
    }


class Pipeline:
    """Holds run state for one trial; supports run / run_until / edit / resume."""

    def __init__(self, request: TargetRequest, *, seed_events=None,
                 providers: dict[str, Callable] | None = None):
        self.request = request
        self.seed_events = seed_events
        self.providers = {**default_providers(), **(providers or {})}
        self.outputs: dict[str, Any] = {}

    def _run_one(self, name: str) -> Any:
        p = self.providers[name]
        if name == "spec":
            out = p(self.request)
        elif name == "plan":
            out = p(self.outputs["spec"], self.request.dataset)
        elif name == "extract":
            out = p(self.outputs["plan"], self.request, self.seed_events)
        elif name == "cohort":
            out = p(self.outputs["extract"], self.outputs["spec"], self.request.dataset)
        elif name == "tte":
            out = p(self.outputs["extract"], self.outputs["cohort"], self.outputs["spec"])
        elif name == "report":
            out = p(self.outputs["tte"], self.request)
        else:  # pragma: no cover
            raise KeyError(name)
        self.outputs[name] = out
        return out

    def run_until(self, name: str) -> Any:
        """Run up to and including `name`, reusing cached upstream outputs."""
        target = STEPS.index(name)
        for n in STEPS[: target + 1]:
            if n not in self.outputs:
                self._run_one(n)
        return self.outputs[name]

    def run(self) -> ComparisonResult:
        return self.run_until("report")

    def get(self, name: str) -> Any:
        return self.outputs.get(name)

    def edit(self, name: str, new_output: Any) -> None:
        """Replace a step's output and invalidate everything downstream
        (selective cache invalidation — the check-&-correct loop)."""
        self.outputs[name] = new_output
        for n in STEPS[STEPS.index(name) + 1:]:
            self.outputs.pop(n, None)

    def resume(self) -> ComparisonResult:
        """Re-run only the invalidated (missing) steps, in order."""
        return self.run()
