"""Batch emulated-vs-observed benchmark (#11, probe / lane:analysis).

Runs the comparison (compare.compare_trial) over many trials x datasets and
aggregates how well the emulations agree with the trials' posted results — the
system's headline deliverable: "how close is the TTE to the real trial, at scale".
"""
from __future__ import annotations

from collections import Counter

from .compare import compare_trial


def run_benchmark(comparisons) -> dict:
    """Aggregate compare_trial() rows into summary metrics. concordance_rate is
    over the COMPARABLE rows only (concordant + discordant; excludes inconclusive
    / not-comparable, which carry no direction)."""
    rows = list(comparisons)
    verdicts = Counter(c.get("verdict") for c in rows)
    comparable = [c for c in rows if c.get("verdict") in ("concordant", "discordant")]
    rate = (sum(c["verdict"] == "concordant" for c in comparable) / len(comparable)
            if comparable else None)
    return {
        "n": len(rows),
        "by_verdict": dict(verdicts),
        "n_comparable": len(comparable),
        "concordance_rate": rate,
    }


def benchmark_trials(items) -> tuple[list[dict], dict]:
    """Run the benchmark over an iterable of (study, tte_result, treatment_hint,
    dataset) tuples. Returns (per-trial rows, summary). One row per trial x DB."""
    rows = [
        compare_trial(study, tte_result, treatment_hint=hint, dataset=dataset)
        for (study, tte_result, hint, dataset) in items
    ]
    return rows, run_benchmark(rows)
