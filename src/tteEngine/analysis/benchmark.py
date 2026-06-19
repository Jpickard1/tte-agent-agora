"""Batch emulated-vs-observed benchmark (#11, probe / lane:analysis).

Aggregates compare.compare_trial rows (contracts.ComparisonResult) over many
trials x datasets into the system's headline metric — how well the emulations
agree with the trials' posted results. Designed to SCALE: run_benchmark consumes
an ITERABLE once and holds only counters (O(#datasets) memory), so it streams
over a >10k-trial corpus.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..contracts.results import Agreement
from .compare import compare_trial

_DIRECTIONAL = (Agreement.CONCORDANT, Agreement.DISCORDANT)


def run_benchmark(comparisons) -> dict:
    """Aggregate an iterable of ComparisonResult -> summary metrics.

    `concordance_rate` is over the COMPARABLE rows only (concordant + discordant;
    excludes inconclusive, which carries no direction). Streams: pass a generator
    of millions of rows and only counters are retained.
    """
    by_agreement: Counter = Counter()
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    n = comparable = concordant = 0
    for c in comparisons:
        n += 1
        by_agreement[c.agreement] += 1
        by_dataset[c.dataset][c.agreement] += 1
        if c.agreement in _DIRECTIONAL:
            comparable += 1
            concordant += int(c.agreement == Agreement.CONCORDANT)
    return {
        "n": n,
        "by_agreement": {a.value: by_agreement[a] for a in Agreement if by_agreement[a]},
        "by_dataset": {ds: {a.value: cc[a] for a in Agreement if cc[a]}
                       for ds, cc in by_dataset.items()},
        "n_comparable": comparable,
        "concordance_rate": (concordant / comparable) if comparable else None,
    }


def benchmark_trials(items) -> tuple[list, dict]:
    """Run the benchmark over an iterable of (study, emulated_TTEResult,
    treatment_hint, dataset). Returns (rows, summary), one row per trial x DB.
    For a very large corpus, stream compare_trial(...) straight into
    run_benchmark instead of materializing rows here."""
    rows = [compare_trial(study, emulated, treatment_hint=hint, dataset=dataset)
            for (study, emulated, hint, dataset) in items]
    return rows, run_benchmark(rows)
