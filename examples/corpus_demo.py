"""#36 corpus batch demo: run the spine over a (synthetic) trial corpus and
print the emulated-vs-observed gallery summary.

Shows the SHAPE of the >1k/>10k run on synthetic data (no ICU data / network
needed). For the real corpus, replace the two injected functions:
  - jobs:       [(study, study_to_spec(study)) for study in fetch_corpus(sepsis_first=True)]
  - extract_fn: adapters.mimic.extract / adapters.eicu.extract on loaded tables
  - engine_fn:  make_engine_provider([... confounders ...], adjustment="iptw")
Everything else (streaming, sepsis-first order, no-silent-cap drop ledger,
benchmark aggregation) is unchanged.

Run:  python examples/corpus_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sepsis_vignette as vig

from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure
from tteEngine.orchestration.corpus import run_corpus_benchmark
from tteEngine.orchestration.pipeline import _crude_rr_engine


def synthetic_jobs(n: int):
    """n synthetic trials (sepsis/steroid spec, distinct NCT ids)."""
    spec = vig.demo_spec()
    return [({"nct": f"NCT{i:05d}"}, spec.model_copy(update={"nct_id": f"NCT{i:05d}"}))
            for i in range(n)]


def _extract(plan, spec, dataset):
    # synthetic stand-in for adapters.mimic/eicu.extract(plan, tables)
    return vig.confounded_stream(scale=1)


def _stub_compare(study, emulated, *, dataset=None):
    same_side = (emulated.estimate - 1.0) * (0.9 - 1.0) > 0
    return ComparisonResult(
        nct_id=emulated.nct_id, dataset=dataset or emulated.dataset, emulated=emulated,
        observed_estimate=0.9, observed_measure=EffectMeasure.RR,
        agreement=Agreement.CONCORDANT if same_side else Agreement.DISCORDANT,
    )


def main(n_trials: int = 25) -> None:
    summary, drops = run_corpus_benchmark(
        synthetic_jobs(n_trials), ["MIMIC-IV", "eICU-CRD"],
        extract_fn=_extract, engine_fn=_crude_rr_engine, compare_fn=_stub_compare,
    )
    print("=" * 64)
    print(f"tteEngine corpus gallery — {n_trials} trials x 2 datasets")
    print("=" * 64)
    print(f"  rows aggregated (streamed): {summary['n']}")
    print(f"  by agreement:               {summary['by_agreement']}")
    print(f"  by dataset:                 {summary['by_dataset']}")
    print(f"  concordance rate:           {summary['concordance_rate']}")
    print(f"  dropped (not silent):       {summary['n_dropped']}  {summary['drops_by_reason']}")
    print("\n  Streaming: memory is O(datasets), so this same call scales to the")
    print("  full >1k/>10k corpus from fetch_corpus(sepsis_first=True).")


if __name__ == "__main__":
    main()
