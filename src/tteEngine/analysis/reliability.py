"""Corpus-level calibration: emulated-vs-observed reliability (#41, probe).

The headline CREDIBILITY metric: across the emulable corpus, how well do the
emulated effect sizes track the trials' actually-reported effects? Produces a
calibration-plot dataset (per-trial emulated vs observed points) + agreement
metrics: calibration slope/intercept (ideal 1 / 0 on the log scale), Pearson r,
CI coverage (fraction of trials whose reported effect falls in the emulated CI),
and RMSE vs the identity line.

Reader-agnostic: takes an Iterable[ComparisonResult] (the #36 corpus, live stream
or contracts.io JSONL). Ratio measures (OR/HR/RR) are compared on the log scale;
risk differences linearly. Pure (numpy/math), no analysis extra.
"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field

from ..contracts.results import EffectMeasure

_RATIO = {EffectMeasure.OR, EffectMeasure.HR, EffectMeasure.RR}


class CalibrationPoint(BaseModel):
    nct_id: str
    dataset: str
    emulated: float
    observed: float
    in_ci: bool          # observed effect falls within the emulated CI


class CalibrationCurve(BaseModel):
    n: int = 0
    scale: str = "log-ratio"
    points: list[CalibrationPoint] = Field(default_factory=list)
    slope: float | None = None        # OLS of observed ~ emulated (ideal 1.0)
    intercept: float | None = None    # ideal 0.0
    pearson_r: float | None = None
    coverage: float | None = None     # fraction of observed within emulated CI
    rmse: float | None = None         # deviation from the identity line
    note: str = ""


def _ols(xs, ys) -> tuple[float, float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 0.0
    intercept = my - slope * mx
    r = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else 0.0
    return slope, intercept, r


def corpus_calibration(comparisons) -> CalibrationCurve:
    """Emulated-vs-observed calibration over the corpus. Uses comparisons where
    both the emulated and observed effects are RATIO measures (log scale)."""
    pts: list[CalibrationPoint] = []
    xs: list[float] = []
    ys: list[float] = []
    covered = 0
    for c in comparisons:
        em, obs, om = c.emulated, c.observed_estimate, c.observed_measure
        if obs is None or em.estimate is None or em.measure not in _RATIO or om not in _RATIO:
            continue
        if em.estimate <= 0 or obs <= 0:
            continue
        in_ci = (em.ci_low is not None and em.ci_high is not None
                 and em.ci_low <= obs <= em.ci_high)
        pts.append(CalibrationPoint(nct_id=c.nct_id, dataset=c.dataset,
                                    emulated=em.estimate, observed=obs, in_ci=in_ci))
        xs.append(math.log(em.estimate))
        ys.append(math.log(obs))
        covered += int(in_ci)

    n = len(xs)
    if n == 0:
        return CalibrationCurve(n=0, note="no ratio-measure comparisons with reported effects")
    coverage = covered / n
    slope = intercept = r = rmse = None
    if n >= 2:
        slope, intercept, r = _ols(xs, ys)
        rmse = math.sqrt(sum((y - x) ** 2 for x, y in zip(xs, ys)) / n)  # vs identity
    return CalibrationCurve(
        n=n, points=pts, slope=slope, intercept=intercept, pearson_r=r,
        coverage=coverage, rmse=rmse,
        note=(f"{n} trials; calibration slope "
              f"{slope if slope is None else round(slope, 2)} (ideal 1.0), "
              f"CI coverage {round(coverage, 2)}."),
    )
