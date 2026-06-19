"""WHY-context panel for the #49 gallery (#98).

Joins worker1's per-(nct_id,dataset) TrialDatasetContext sidecar (#95/#96) onto
the gallery rows and distills it into render-ready "why" summaries: why a trial
is emulable in a dataset, which protocol elements are measured / proxied /
unmeasurable, and why datasets diverge (#32 attribution). PURE / import-light
(reads the contracts.context records only — no analysis, no matplotlib), so #49
renders it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from tteEngine.contracts.context import TrialDatasetContext


def index_context(records: Iterable["TrialDatasetContext"]) -> dict[tuple[str, str], "TrialDatasetContext"]:
    """Index the sidecar by the (nct_id, dataset) join key — same key as the
    corpus ComparisonResult rows / Trial Emulation Cards."""
    return {(r.nct_id, r.dataset): r for r in records}


def _proxy_elements(ctx) -> list[str]:
    out = []
    for p in ctx.proxy_list or []:
        name = p.get("element") or p.get("name") or p.get("concept") if isinstance(p, dict) else None
        out.append(str(name) if name else str(p))
    return out


def why_for(ctx: "TrialDatasetContext") -> dict:
    """Render-ready 'why' for one (trial, dataset) row."""
    m = ctx.measurability or {}
    gaps = m.get("gaps") or []
    if ctx.emulable:
        why_emulable = f"Emulable in {ctx.dataset} (score {ctx.emulability_score:.2f})."
    else:
        gap_txt = f" — gaps: {', '.join(map(str, gaps[:3]))}" if gaps else ""
        why_emulable = f"Not emulable in {ctx.dataset} (score {ctx.emulability_score:.2f}){gap_txt}."

    why_divergent = None
    if ctx.variability:
        attr = ctx.variability.get("attribution") or {}
        het = ctx.variability.get("heterogeneity") or {}
        note = attr.get("note")
        causes = attr.get("causes") or []
        i2 = het.get("i2")
        bits = []
        if i2 is not None:
            bits.append(f"I²={i2:.0%}" if isinstance(i2, (int, float)) else f"I²={i2}")
        if causes:
            bits.append("drivers: " + ", ".join(map(str, causes[:3])))
        if note:
            bits.append(str(note))
        why_divergent = "; ".join(bits) if bits else None

    return {
        "nct_id": ctx.nct_id,
        "dataset": ctx.dataset,
        "is_sepsis": ctx.is_sepsis,
        "emulable": ctx.emulable,
        "emulability_score": ctx.emulability_score,
        "why_emulable": why_emulable,
        "measurability": {
            "n_measurable": m.get("n_measurable"),
            "n_proxy": m.get("n_proxy"),
            "n_unmeasurable": m.get("n_unmeasurable"),
            "fully_measurable": m.get("fully_measurable"),
            "gaps": gaps,
        },
        "proxy_elements": _proxy_elements(ctx),
        "missingness": ctx.missingness,
        "why_divergent": why_divergent,
    }


def corpus_context_summary(records: Iterable["TrialDatasetContext"]) -> dict:
    """Corpus-level measurability/emulability rollup for the gallery header."""
    recs = list(records)
    n = len(recs)
    if n == 0:
        return {"n": 0}
    n_emulable = sum(1 for r in recs if r.emulable)
    fully = sum(1 for r in recs if (r.measurability or {}).get("fully_measurable"))
    scores = [r.emulability_score for r in recs if r.emulability_score is not None]
    proxy_counts: dict[str, int] = {}
    for r in recs:
        for el in _proxy_elements(r):
            proxy_counts[el] = proxy_counts.get(el, 0) + 1
    top_proxy = sorted(proxy_counts.items(), key=lambda kv: -kv[1])[:5]
    return {
        "n": n,
        "n_emulable": n_emulable,
        "pct_emulable": n_emulable / n,
        "pct_fully_measurable": fully / n,
        "mean_emulability_score": (sum(scores) / len(scores)) if scores else None,
        "n_sepsis": sum(1 for r in recs if r.is_sepsis),
        "top_proxy_elements": [{"element": e, "n": c} for e, c in top_proxy],
    }
