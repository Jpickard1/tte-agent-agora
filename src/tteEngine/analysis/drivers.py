"""Concordance-driver analysis — "find the story" (#61, probe / lane:analysis).

Turns the corpus of emulated-vs-observed comparisons into the journal narrative:
WHICH trial/data features predict CONCORDANCE vs discordance — endpoint type,
effect-size magnitude, sample size, precision, dataset, and (via optional joins)
follow-up length, measurability, and sepsis. For each feature it reports the
concordance rate per stratum and the spread (how strongly the feature separates
agree from disagree), ranks features by that spread, and calls out sepsis.

Reader-agnostic: takes Iterable[ComparisonResult] (the saved #36 corpus via
contracts.io.load_comparisons_jsonl, or the live stream). Pure (numpy/math).
write_narrative() emits the RESULTS_NARRATIVE.md headline claims + numbers.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable

from pydantic import BaseModel, Field

from ..contracts.results import Agreement, EffectMeasure

_RATIO = {EffectMeasure.OR, EffectMeasure.HR, EffectMeasure.RR}
_ENDPOINT = {"HR": "survival", "OR": "binary", "RR": "binary", "RD": "risk"}


def extract_features(c, *, follow_up_fn: Callable | None = None,
                     measurability_fn: Callable | None = None,
                     sepsis_fn: Callable | None = None) -> dict:
    """Feature vector for one comparison: derived from the ComparisonResult, plus
    optional joins (follow-up from the spec, measurability from #33, sepsis from
    the catalog) supplied as callables so #61 stays self-contained."""
    em = c.emulated
    f: dict = {"dataset": c.dataset or "?", "endpoint_type": _ENDPOINT.get(em.measure.value, "other")}
    if em.estimate is not None:
        if em.measure in _RATIO and em.estimate > 0:
            f["effect_magnitude"] = abs(math.log(em.estimate))
        elif em.measure not in _RATIO:
            f["effect_magnitude"] = abs(em.estimate)
    f["n_total"] = (em.n_treated or 0) + (em.n_control or 0)
    if em.measure in _RATIO and em.ci_low and em.ci_high and em.ci_low > 0:
        f["ci_width_log"] = math.log(em.ci_high) - math.log(em.ci_low)
    ev = (em.extra or {}).get("e_value_point")
    if ev is not None:
        f["e_value"] = ev
    if follow_up_fn is not None:
        fu = follow_up_fn(c.nct_id)
        if fu is not None:
            f["follow_up_hours"] = fu
    if measurability_fn is not None:
        ms = measurability_fn(c.nct_id, c.dataset)
        if ms is not None:
            f["measurability"] = ms
    if sepsis_fn is not None:
        f["is_sepsis"] = bool(sepsis_fn(c))
    return f


class FeatureAssociation(BaseModel):
    feature: str
    kind: str                      # 'categorical' | 'continuous'
    strata: list[dict] = Field(default_factory=list)  # [{level, n, n_concordant, concordance_rate}]
    spread: float | None = None    # max-min concordance rate across strata (association strength)


class DriverReport(BaseModel):
    n: int = 0
    n_comparable: int = 0
    overall_concordance: float | None = None
    associations: list[FeatureAssociation] = Field(default_factory=list)  # ranked by spread
    sepsis_finding: str | None = None


def _strata_categorical(vals) -> list[dict]:
    g: dict[str, list] = defaultdict(list)
    for v, conc in vals:
        g[str(v)].append(conc)
    return [{"level": k, "n": len(c), "n_concordant": sum(c),
             "concordance_rate": sum(c) / len(c)} for k, c in sorted(g.items())]


def _strata_median_split(vals) -> list[dict]:
    xs = sorted(v for v, _ in vals)
    med = xs[len(xs) // 2]
    lo = [conc for v, conc in vals if v < med]
    hi = [conc for v, conc in vals if v >= med]
    out = []
    for label, c in ((f"< {med:.3g}", lo), (f">= {med:.3g}", hi)):
        if c:
            out.append({"level": label, "n": len(c), "n_concordant": sum(c),
                        "concordance_rate": sum(c) / len(c)})
    return out


def concordance_drivers(comparisons, *, follow_up_fn=None, measurability_fn=None,
                        sepsis_fn=None) -> DriverReport:
    """Which features predict concordance. Over the COMPARABLE rows (concordant +
    discordant), stratify each feature and rank by concordance-rate spread."""
    rows = [c for c in comparisons
            if c.agreement in (Agreement.CONCORDANT, Agreement.DISCORDANT)]
    table = []
    for c in rows:
        feats = extract_features(c, follow_up_fn=follow_up_fn,
                                 measurability_fn=measurability_fn, sepsis_fn=sepsis_fn)
        feats["_concordant"] = c.agreement == Agreement.CONCORDANT
        table.append(feats)
    n = len(table)
    overall = (sum(r["_concordant"] for r in table) / n) if n else None

    names: set[str] = set()
    for r in table:
        names.update(k for k in r if not k.startswith("_") and r[k] is not None)
    assocs: list[FeatureAssociation] = []
    for feat in sorted(names):
        vals = [(r[feat], r["_concordant"]) for r in table if r.get(feat) is not None]
        if len({v for v, _ in vals}) < 2:   # need >=2 distinct values to stratify
            continue
        if isinstance(vals[0][0], (str, bool)):
            strata, kind = _strata_categorical(vals), "categorical"
        else:
            strata, kind = _strata_median_split(vals), "continuous"
        rates = [s["concordance_rate"] for s in strata if s["n"] > 0]
        spread = (max(rates) - min(rates)) if len(rates) >= 2 else None
        assocs.append(FeatureAssociation(feature=feat, kind=kind, strata=strata, spread=spread))
    assocs.sort(key=lambda a: (a.spread is not None, a.spread or 0.0), reverse=True)

    sepsis_finding = None
    if sepsis_fn is not None:
        sep = [r["_concordant"] for r in table if r.get("is_sepsis") is True]
        non = [r["_concordant"] for r in table if r.get("is_sepsis") is False]
        if sep:
            sr = sum(sep) / len(sep)
            nr = (sum(non) / len(non)) if non else None
            sepsis_finding = (f"Sepsis concordance {sr:.0%} (n={len(sep)})"
                              + (f" vs non-sepsis {nr:.0%} (n={len(non)})." if nr is not None else "."))
    return DriverReport(n=n, n_comparable=n, overall_concordance=overall,
                        associations=assocs, sepsis_finding=sepsis_finding)


def write_narrative(drivers: DriverReport, *, meta=None, calibration=None) -> str:
    """Generate RESULTS_NARRATIVE.md (headline claims + supporting numbers) from
    the driver report (+ optional #64 meta + #41 calibration)."""
    L = ["# TTE Emulation — Results Narrative", "",
         "_Auto-generated from the emulated-vs-observed corpus._", ""]
    L.append("## Headline")
    if meta is not None and meta.overall_concordance.rate is not None:
        oc = meta.overall_concordance
        L.append(f"- Across {oc.n_comparable} comparable trial-emulations, the emulation reproduced "
                 f"the real RCT's direction in **{oc.rate:.0%}** (95% CI "
                 f"{oc.ci_low:.0%}–{oc.ci_high:.0%}).")
    elif drivers.overall_concordance is not None:
        L.append(f"- Overall concordance: **{drivers.overall_concordance:.0%}** "
                 f"over {drivers.n_comparable} comparable emulations.")
    if calibration is not None and calibration.slope is not None:
        L.append(f"- Calibration slope **{calibration.slope:.2f}** (ideal 1.0), "
                 f"CI coverage **{calibration.coverage:.0%}** over {calibration.n} trials.")
    if meta is not None and meta.pooled_effect.i2 is not None:
        L.append(f"- Between-trial heterogeneity I² = **{meta.pooled_effect.i2:.0f}%**.")
    if drivers.sepsis_finding:
        L.append(f"- **Sepsis:** {drivers.sepsis_finding}")
    L += ["", "## What predicts concordance (ranked drivers)"]
    for a in drivers.associations[:8]:
        sp = "n/a" if a.spread is None else f"{a.spread:.0%} rate spread"
        L.append(f"- **{a.feature}** ({a.kind}, {sp}): "
                 + "; ".join(f"{s['level']}={s['concordance_rate']:.0%} (n={s['n']})" for s in a.strata))
    L += ["", "## Where emulation succeeds / fails",
          "- Strata with the highest concordance mark where TTE is trustworthy; the lowest "
          "mark failure modes (residual confounding, outcome-proxy mismatch, sparse data).",
          "", "_Numbers populate from the live MIMIC/eICU corpus run; regenerate via "
          "`write_narrative(concordance_drivers(load_comparisons_jsonl(corpus)), ...)`._"]
    return "\n".join(L)
