"""Clinician Trial Emulation Card (#40) — the per-trial unit of the #49 gallery.

One card per (trial x dataset) ComparisonResult: the emulated effect vs the real
RCT's reported effect, the agreement verdict in plain language, and the key
diagnostics a clinician scans. PURE / import-light (operates on ComparisonResult +
.extra), so the UI renders it without matplotlib or the analysis extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from pydantic import BaseModel

if TYPE_CHECKING:
    from tteEngine.contracts.results import ComparisonResult

_RATIO = {"RR", "OR", "HR"}


def _measure(e) -> str:
    return e.measure.value if hasattr(e.measure, "value") else str(e.measure)


def _agree(c) -> str:
    return c.agreement.value if hasattr(c.agreement, "value") else str(c.agreement)


def _direction(estimate: float, measure: str) -> str:
    null = 1.0 if measure in _RATIO else 0.0
    if estimate == null:
        return "no effect"
    return "benefit" if estimate < null else "harm"


def _verdict(c) -> str:
    """Plain-language reading a clinician can scan."""
    a = _agree(c)
    if a == "concordant":
        d = _direction(c.emulated.estimate, _measure(c.emulated))
        return f"Emulation AGREES with the trial ({d})."
    if a == "discordant":
        return "Emulation DISAGREES with the trial (opposite direction)."
    return "Inconclusive — no clear direction or no reported trial effect."


class TrialEmulationCard(BaseModel):
    nct_id: str
    dataset: str
    is_sepsis: bool = False
    measure: str
    emulated_estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    observed_estimate: float | None = None
    agreement: str
    verdict: str
    p_value: float | None = None
    e_value: float | None = None
    n_treated: int = 0
    n_control: int = 0
    why: dict | None = None  # WHY-context (#98), joined from worker1's sidecar by (nct_id, dataset)


def build_cards(
    comparisons: Iterable["ComparisonResult"],
    *,
    sepsis_ncts: set[str] | None = None,
) -> list[TrialEmulationCard]:
    """One card per comparison (streams). `sepsis_ncts` (e.g. from the #35 catalog)
    marks sepsis trials so the gallery can filter/prioritize them."""
    sepsis = sepsis_ncts or set()
    cards: list[TrialEmulationCard] = []
    for c in comparisons:
        e = c.emulated
        extra = e.extra or {}
        cards.append(TrialEmulationCard(
            nct_id=c.nct_id, dataset=c.dataset, is_sepsis=c.nct_id in sepsis,
            measure=_measure(e), emulated_estimate=e.estimate,
            ci_low=e.ci_low, ci_high=e.ci_high, observed_estimate=c.observed_estimate,
            agreement=_agree(c), verdict=_verdict(c),
            p_value=extra.get("p_value"), e_value=extra.get("e_value_point"),
            n_treated=e.n_treated, n_control=e.n_control,
        ))
    return cards
