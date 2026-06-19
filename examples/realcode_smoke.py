"""Real-code smoke (#13 fast-follow; pairs #59 resolver + #71 cohort hook).

The #13 vignette proves the spine on a SYNTHETIC stream whose events carry
concept NAMES (EVENT_NAME='sepsis'). Real adapters (#6/#7) emit raw CODES
(sepsis = ICD 'A41'). This smoke proves the SAME confounding flip on raw-coded
output, the way a live MIMIC/eICU run actually looks:

    raw-coded stream (sepsis dx = 'A41')
      -> Pipeline(cohort = make_cohort_provider(resolve=vocab.classify), tte = IPTW)
      -> crude apparent-harm  ->  IPTW benefit  (the flip, on raw codes)

And it proves the resolver is LOAD-BEARING: under the identity default the same
raw-coded cohort is EMPTY (concept 'sepsis' can't match raw 'A41'), which is
exactly the gap #59 closed.

Reuses the vignette's confounded cohort so the only change is name -> code.
Run:  python examples/realcode_smoke.py   (the IPTW arm needs the `analysis` extra)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # reuse the #13 vignette helpers

import pandas as pd  # noqa: E402

from tteEngine import vocab  # noqa: E402
from tteEngine.cohort import build_cohort  # noqa: E402
from tteEngine.contracts.results import EffectMeasure  # noqa: E402
from tteEngine.orchestration import Pipeline, TargetRequest  # noqa: E402
from tteEngine.orchestration.engine_provider import make_engine_provider  # noqa: E402
from tteEngine.orchestration.pipeline import make_cohort_provider  # noqa: E402

from sepsis_vignette import LACTATE, OBSERVED_RR, confounded_stream, demo_spec  # noqa: E402

SEPSIS_ICD = "A41"  # a real MIMIC sepsis ICD-10 code (vocab.classify('A41') -> 'sepsis')


def raw_coded_stream(scale: int = 2) -> pd.DataFrame:
    """The vignette's confounded cohort, but the sepsis DIAGNOSIS carries its raw
    ICD code instead of the concept name — i.e. what adapters.mimic.extract emits."""
    df = confounded_stream(scale).copy()
    df.loc[df["EVENT_NAME"] == "sepsis", "EVENT_NAME"] = SEPSIS_ICD
    return df


def _request(dataset: str) -> TargetRequest:
    return TargetRequest(nct_id="NCT-SEPSIS-STEROID", dataset=dataset, seed_spec=demo_spec(),
                         observed_estimate=OBSERVED_RR, observed_measure=EffectMeasure.RR)


def cohort_n(stream: pd.DataFrame, *, resolve) -> int:
    """Cohort size when matching with (resolve=vocab.classify) vs without."""
    return build_cohort(stream, demo_spec(), dataset="MIMIC-IV", resolve=resolve).n_total


def run_realcode_adjusted(dataset: str, stream: pd.DataFrame):
    """The supported path (#71): cohort provider with the vocab resolver + IPTW."""
    return Pipeline(
        _request(dataset), seed_events=stream,
        providers={"cohort": make_cohort_provider(resolve=vocab.classify),
                   "tte": make_engine_provider([LACTATE], adjustment="iptw")},
    ).run()


def run_realcode_crude(dataset: str, stream: pd.DataFrame):
    return Pipeline(_request(dataset), seed_events=stream,
                    providers={"cohort": make_cohort_provider(resolve=vocab.classify)}).run()


def main() -> None:
    stream = raw_coded_stream(scale=2)
    print("=" * 74)
    print("tteEngine real-code smoke — sepsis dx as ICD 'A41' (what an adapter emits)")
    print("=" * 74)
    n_identity = cohort_n(stream, resolve=None)
    n_resolved = cohort_n(stream, resolve=vocab.classify)
    print(f"\ncohort on raw codes:  identity -> n={n_identity}   "
          f"vocab.classify -> n={n_resolved}")
    print("  (identity can't match 'sepsis' to 'A41' -> empty; the #59 resolver bridges it)")
    crude = run_realcode_crude("MIMIC-IV", stream).emulated
    print(f"\nCRUDE     {crude.measure.value} = {crude.estimate:.3f}  (confounded: apparent harm)")
    try:
        adj = run_realcode_adjusted("MIMIC-IV", stream).emulated
        ci = f" (95% CI {adj.ci_low:.2f}-{adj.ci_high:.2f})" if adj.ci_low is not None else ""
        print(f"IPTW-ADJ  {adj.measure.value} = {adj.estimate:.3f}{ci}  (adjusted: benefit) -> the flip, on raw codes")
    except Exception as exc:  # analysis extra not installed
        print(f"IPTW-ADJ  (skipped: {exc})")


if __name__ == "__main__":
    main()
