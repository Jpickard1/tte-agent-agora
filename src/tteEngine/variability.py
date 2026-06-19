"""Cross-dataset variability explainer / heterogeneity (#32, worker1).

For one trial emulated across MIMIC / eICU / MGB, this answers jpic's
explain-the-variability ask: it compares the per-dataset effect, quantifies
heterogeneity (I²/τ²/Q), lays out forest-plot rows, and — the value-add over a
bare meta-analysis — ATTRIBUTES the divergence to concrete causes:

  * cohort       — the per-dataset cohort sizes differ (small-n / different pops);
  * coding/measurability — a protocol element is measurable in one dataset but
                   only a proxy/unmeasurable in another (from #33);
  * missingness  — an adjustment variable is far more missing in one dataset
                   (from #34, when analysis frames are supplied).

It REUSES probe's #64 meta engine for the heterogeneity math (random_effects →
I²/τ²) rather than reimplementing it — imported lazily so this module stays
import-light for CI's [dev] env. The attribution layer is pure (built on #33
measurability + cohort sizes; #34 missingness is optional).

Acceptance (#32): per-trial forest rows + a heterogeneity-attribution note.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contracts.trial_spec import TargetTrialSpec
from .measurability import MEASURABLE, measurability_report

# below this min/max cohort-size ratio we call the cohorts materially different
_COHORT_SPREAD = 0.5
# adjustment-missingness gap (fraction) that counts as a divergence driver
_MISSING_GAP = 0.2


def forest_rows(results) -> list[dict]:
    """Per-dataset forest-plot rows for one trial (the figure input)."""
    return [{
        "nct_id": r.nct_id, "dataset": r.dataset, "measure": r.emulated.measure.value,
        "estimate": r.emulated.estimate, "ci_low": r.emulated.ci_low,
        "ci_high": r.emulated.ci_high,
        "n": r.emulated.n_treated + r.emulated.n_control,
        "n_treated": r.emulated.n_treated, "n_control": r.emulated.n_control,
    } for r in results]


def cross_dataset_heterogeneity(results) -> dict:
    """I²/τ²/Q + random-effects pooled estimate over the per-dataset effects
    (reuses #64 random_effects). `results` = one ComparisonResult per dataset."""
    from .analysis.meta import _y_se, random_effects  # lazy: keep import-light

    items = [ys for ys in (_y_se(r.emulated) for r in results) if ys]
    m = random_effects(items, ratio=True)
    return {"k": m.k, "i2": m.i2, "tau2": m.tau2, "q": m.q,
            "pooled_estimate": m.pooled_estimate, "ci_low": m.ci_low, "ci_high": m.ci_high}


@dataclass
class VariabilityAttribution:
    causes: list[dict] = field(default_factory=list)   # {cause, detail}
    note: str = ""


def _cohort_cause(results) -> dict | None:
    ns = {r.dataset: r.emulated.n_treated + r.emulated.n_control for r in results}
    sizes = [n for n in ns.values() if n > 0]
    if len(sizes) < 2 or max(sizes) == 0:
        return None
    if min(sizes) / max(sizes) < _COHORT_SPREAD:
        detail = ", ".join(f"{d} n={n}" for d, n in ns.items())
        return {"cause": "cohort", "detail": f"cohort sizes differ materially ({detail})"}
    return None


def _measurability_causes(spec: TargetTrialSpec, datasets: list[str]) -> list[dict]:
    """Elements whose measurability status differs across the datasets (#33)."""
    reports = {ds: measurability_report(spec, ds) for ds in datasets}
    # key elements by (kind, concept); compare status across datasets
    keyed: dict[tuple, dict[str, tuple[str, str]]] = {}
    for ds, rep in reports.items():
        for e in rep.elements:
            keyed.setdefault((e.kind, e.concept), {})[ds] = (e.status, e.reason)
    causes: list[dict] = []
    for (kind, concept), by_ds in keyed.items():
        statuses = {v[0] for v in by_ds.values()}
        if len(statuses) > 1:  # divergent element
            spread = "; ".join(f"{ds}={st}" for ds, (st, _) in by_ds.items())
            causes.append({"cause": "coding/measurability",
                           "detail": f"{kind} '{concept}' differs by dataset ({spread})"})
    return causes


def _missingness_causes(frames: dict, columns: list[str] | None) -> list[dict]:
    """Adjustment variables whose missing fraction differs a lot across datasets (#34)."""
    from .missingness import missingness_summary

    summaries = {ds: missingness_summary(fr, columns) for ds, fr in frames.items()}
    causes: list[dict] = []
    feats: set[str] = set()
    for s in summaries.values():
        feats |= set(s["columns"])
    for f in sorted(feats):
        fracs = {ds: s["columns"].get(f, {}).get("missing_fraction") for ds, s in summaries.items()}
        present = {ds: v for ds, v in fracs.items() if v is not None}
        if len(present) >= 2 and (max(present.values()) - min(present.values())) >= _MISSING_GAP:
            spread = ", ".join(f"{ds}={v:.0%}" for ds, v in present.items())
            causes.append({"cause": "missingness",
                           "detail": f"'{f}' missingness differs across datasets ({spread})"})
    return causes


def attribute_variability(
    spec: TargetTrialSpec, results, *, frames: dict | None = None,
    feature_columns: list[str] | None = None,
) -> VariabilityAttribution:
    """Attribute cross-dataset divergence to cohort / coding-measurability /
    missingness causes (the heterogeneity-attribution note)."""
    datasets = [r.dataset for r in results]
    causes: list[dict] = []
    cohort = _cohort_cause(results)
    if cohort:
        causes.append(cohort)
    causes.extend(_measurability_causes(spec, datasets))
    if frames:
        causes.extend(_missingness_causes(frames, feature_columns))

    if not causes:
        note = (f"Estimates across {', '.join(datasets)} are not attributably divergent: "
                "cohorts comparable, all elements equally measurable"
                + (", missingness similar" if frames else "") + ".")
    else:
        labels = sorted({c["cause"] for c in causes})
        note = (f"Cross-dataset divergence for {spec.nct_id} attributable to "
                f"{', '.join(labels)}. " + " ".join(f"[{c['cause']}] {c['detail']}." for c in causes))
    return VariabilityAttribution(causes=causes, note=note)


def variability_report(
    spec: TargetTrialSpec, results, *, frames: dict | None = None,
    feature_columns: list[str] | None = None,
) -> dict:
    """The #32 report: forest rows + heterogeneity (I²/τ²) + attribution note,
    for one trial emulated across datasets. `results` = one ComparisonResult per
    dataset; `frames` (optional) = {dataset: analysis_frame} for missingness attribution."""
    attr = attribute_variability(spec, results, frames=frames, feature_columns=feature_columns)
    return {
        "nct_id": spec.nct_id,
        "datasets": [r.dataset for r in results],
        "forest": forest_rows(results),
        "heterogeneity": cross_dataset_heterogeneity(results),
        "attribution": {"causes": attr.causes, "note": attr.note},
    }


__all__ = [
    "forest_rows", "cross_dataset_heterogeneity", "attribute_variability",
    "variability_report", "VariabilityAttribution",
]
