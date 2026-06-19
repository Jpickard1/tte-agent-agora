"""Corpus persistence (JSONL) — the saved #36 artifact, import-light (#64 follow-up).

Lives in contracts/ (next to ComparisonResult, no heavy deps) so the orchestration
can persist the corpus stream WITHOUT importing analysis — preserving the
graceful-degradation seam. One source of truth read/written by #36 (producer),
#64 meta-analysis, figures (#60), and the UI (#49).

Schema: one ``ComparisonResult.model_dump_json()`` per line. Streaming (O(1) memory),
safe for a >10k-trial corpus.
"""
from __future__ import annotations

from .results import ComparisonResult


def dump_comparisons_jsonl(comparisons, path) -> int:
    """Persist a ComparisonResult stream to JSONL (one model_dump_json per line).
    Streams; returns the count written."""
    n = 0
    with open(path, "w") as f:
        for c in comparisons:
            f.write(c.model_dump_json() + "\n")
            n += 1
    return n


def load_comparisons_jsonl(path):
    """Iterate ComparisonResult from a JSONL dump (lazy — streams into meta_analyze)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield ComparisonResult.model_validate_json(line)
