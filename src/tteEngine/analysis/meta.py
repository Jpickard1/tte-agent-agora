"""Cross-trial meta-analysis + heterogeneity (#64, flagship — probe / lane:analysis).

The paper's headline quantitative result. Over the corpus of emulated-vs-observed
comparisons (#11 ComparisonResults from the #36 corpus run), it reports:
- overall CONCORDANCE RATE + Wilson CI (the top-line stat), and per subgroup;
- a random-effects (DerSimonian–Laird) META-ANALYSIS of the emulated effect sizes
  with HETEROGENEITY (I^2, tau^2, Cochran's Q), overall + sepsis subgroup;
- forest-plot rows (per-trial estimate + CI + agreement) for the figure.

Pure (numpy/math); no analysis extra needed — it consumes the (estimate, CI)
pairs the results already carry. Ratio measures (OR/HR/RR) are pooled on the log
scale; risk differences on the linear scale.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from pydantic import BaseModel, Field

from ..contracts.results import Agreement, ComparisonResult, EffectMeasure

_RATIO = {EffectMeasure.OR, EffectMeasure.HR, EffectMeasure.RR}
_Z = 1.959963984540054  # 97.5th percentile of the standard normal


def wilson_ci(k: int, n: int, z: float = _Z) -> tuple[float | None, float | None]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return None, None
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


class ConcordanceSummary(BaseModel):
    n: int = 0
    n_comparable: int = 0
    n_concordant: int = 0
    rate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


def concordance_summary(comparisons) -> ConcordanceSummary:
    """Concordance rate + Wilson CI over the COMPARABLE rows (concordant +
    discordant; inconclusive carries no direction)."""
    rows = list(comparisons)
    comp = [c for c in rows if c.agreement in (Agreement.CONCORDANT, Agreement.DISCORDANT)]
    k = sum(c.agreement == Agreement.CONCORDANT for c in comp)
    lo, hi = wilson_ci(k, len(comp))
    return ConcordanceSummary(n=len(rows), n_comparable=len(comp), n_concordant=k,
                              rate=(k / len(comp) if comp else None), ci_low=lo, ci_high=hi)


def _y_se(tte) -> tuple[float, float] | None:
    """(point, se) on the analysis scale (log for ratio measures). None if unusable."""
    est, lo, hi = tte.estimate, tte.ci_low, tte.ci_high
    if est is None or lo is None or hi is None:
        return None
    if tte.measure in _RATIO:
        if est <= 0 or lo <= 0 or hi <= 0:
            return None
        y, se = math.log(est), (math.log(hi) - math.log(lo)) / (2 * _Z)
    else:
        y, se = est, (hi - lo) / (2 * _Z)
    return (y, se) if math.isfinite(se) and se > 0 else None


class MetaResult(BaseModel):
    k: int = 0
    scale: str = "log-ratio"
    pooled_estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    tau2: float | None = None
    i2: float | None = None
    q: float | None = None
    note: str = ""


def random_effects(items: list[tuple[float, float]], *, ratio: bool = True) -> MetaResult:
    """DerSimonian–Laird random-effects pool of (y, se) pairs (y on the analysis
    scale). Returns pooled estimate + CI (back-transformed for ratios), tau^2, I^2, Q."""
    scale = "log-ratio" if ratio else "linear"
    k = len(items)
    if k == 0:
        return MetaResult(k=0, scale=scale, note="no usable estimates")
    ys = [y for y, _ in items]
    ses = [se for _, se in items]
    w = [1 / (se * se) for se in ses]
    sw = sum(w)
    ybar = sum(wi * yi for wi, yi in zip(w, ys)) / sw
    q = sum(wi * (yi - ybar) ** 2 for wi, yi in zip(w, ys))
    df = k - 1
    c = sw - sum(wi * wi for wi in w) / sw
    tau2 = max(0.0, (q - df) / c) if c > 0 and df > 0 else 0.0
    wr = [1 / (se * se + tau2) for se in ses]
    swr = sum(wr)
    pooled = sum(wi * yi for wi, yi in zip(wr, ys)) / swr
    se_p = math.sqrt(1 / swr)
    lo, hi = pooled - _Z * se_p, pooled + _Z * se_p
    i2 = max(0.0, (q - df) / q) * 100 if (q > 0 and df > 0) else 0.0
    if ratio:
        pooled, lo, hi = math.exp(pooled), math.exp(lo), math.exp(hi)
    return MetaResult(k=k, scale=scale, pooled_estimate=pooled, ci_low=lo, ci_high=hi,
                      tau2=tau2, i2=i2, q=q)


def pooled_effect(comparisons) -> MetaResult:
    """Random-effects pool of emulated RATIO-measure (OR/HR/RR) estimates."""
    items = [yse for c in comparisons
             if c.emulated.measure in _RATIO and (yse := _y_se(c.emulated)) is not None]
    return random_effects(items, ratio=True)


class SubgroupMeta(BaseModel):
    name: str
    concordance: ConcordanceSummary
    pooled_effect: MetaResult


class MetaReport(BaseModel):
    overall_concordance: ConcordanceSummary
    pooled_effect: MetaResult
    by_subgroup: list[SubgroupMeta] = Field(default_factory=list)
    forest: list[dict] = Field(default_factory=list)


def meta_analyze(comparisons, *, subgroup: Callable[[ComparisonResult], str] | None = None
                 ) -> MetaReport:
    """The flagship roll-up: overall concordance rate + CI, random-effects pooled
    effect + heterogeneity, optional subgroup breakdown (e.g. sepsis), forest rows."""
    rows = list(comparisons)
    forest = [{"nct_id": c.nct_id, "dataset": c.dataset, "measure": c.emulated.measure.value,
               "estimate": c.emulated.estimate, "ci_low": c.emulated.ci_low,
               "ci_high": c.emulated.ci_high, "agreement": c.agreement.value} for c in rows]
    report = MetaReport(overall_concordance=concordance_summary(rows),
                        pooled_effect=pooled_effect(rows), forest=forest)
    if subgroup is not None:
        groups: dict[str, list] = defaultdict(list)
        for c in rows:
            groups[subgroup(c)].append(c)
        for name in sorted(groups):
            g = groups[name]
            report.by_subgroup.append(SubgroupMeta(
                name=name, concordance=concordance_summary(g), pooled_effect=pooled_effect(g)))
    return report


# --- corpus persistence (JSONL) ------------------------------------------------
# #36 streams ComparisonResults in-memory; these materialize the stream ONCE so
# #64 (this), figures (#60) and the UI (#49) read the saved corpus offline (no
# re-extraction). Schema = one ComparisonResult.model_dump_json() per line.

def dump_comparisons_jsonl(comparisons, path) -> int:
    """Persist a ComparisonResult stream to JSONL (one model_dump_json per line).
    Streams — safe for a >10k corpus. Returns the count written."""
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
