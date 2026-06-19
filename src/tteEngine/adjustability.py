"""Confounder adjustability ledger + PS diagnostics builder (#105, probe).

Makes the considered / adjustable / not-adjustable confounder split EXPLICIT, per
(trial, dataset): take the CONSIDERED set (a standard ICU confounder set), classify
each as ADJUSTED (measurable here AND in the model), MEASURABLE-not-used, or
NOT-ADJUSTABLE (unmeasurable/proxy-only here -> residual confounding), join the
covariates actually used (engine output, surfaced in TTEResult.extra) with worker1's
#33 measurability, attach the SMD balance + PS-overlap diagnostic, and tie the
not-adjustable set to the #37 E-value residual-confounding narrative.

Persisted as ledger.jsonl alongside corpus.jsonl / context.jsonl (same
(nct_id, dataset) key), so the #104 UI + meta-analysis read it. Import-light:
measurability is pure; no analysis/heavy deps at module load.
"""
from __future__ import annotations

from .contracts.adjustability import (
    Adjustability,
    ConfounderLedger,
    ConfounderRow,
    dump_ledger_jsonl,
)
from .contracts.events import EventType

#: The prespecified CONSIDERED adjustment set: canonical ICU confounders, each with
#: its event domain + name aliases used to match model covariate columns. (concept,
#: event_type, aliases). Extend per-trial from baseline characteristics as a follow-up.
STANDARD_ICU_CONFOUNDERS: list[tuple[str, EventType, tuple[str, ...]]] = [
    ("age", EventType.DEMOGRAPHIC, ("age",)),
    ("sex", EventType.DEMOGRAPHIC, ("sex", "gender")),
    ("severity score (SOFA/APACHE)", EventType.MEASUREMENT, ("sofa", "apache", "saps", "severity")),
    ("lactate", EventType.LAB, ("lactate", "lactic")),
    ("creatinine", EventType.LAB, ("creatinine", "creat")),
    ("bilirubin", EventType.LAB, ("bilirubin",)),
    ("platelets", EventType.LAB, ("platelet", "plt")),
    ("white blood cell count", EventType.LAB, ("wbc", "white blood", "leukocyte")),
    ("mean arterial pressure", EventType.MEASUREMENT, ("map", "mean arterial", "mbp", "blood pressure")),
    ("heart rate", EventType.MEASUREMENT, ("heart rate", "heartrate", "heart_rate")),
    ("vasopressor use", EventType.MEDICATION, ("vasopressor", "norepinephrine", "vasopressin", "pressor")),
    ("mechanical ventilation", EventType.PROCEDURE, ("ventilation", "ventilator", "intubat", "mech vent")),
    ("comorbidity burden", EventType.DIAGNOSIS, ("comorbid", "charlson", "elixhauser")),
]


def _default_measure_fn(concept: str, event_type: EventType, dataset: str) -> tuple[str, str]:
    """Per-confounder measurability verdict (status, reason) via worker1's public
    #117 entry. Injectable so the verdict source can be swapped/refined without
    touching this builder."""
    from .measurability import confounder_measurability

    d = confounder_measurability(concept, event_type, dataset)
    return d["status"], d["reason"]


def _alias_match(name: str, aliases: tuple[str, ...]) -> bool:
    n = (name or "").lower().replace("_", " ")
    return any(al in n for al in aliases)


def _smd_for(aliases, balance_rows) -> tuple[float | None, float | None]:
    for row in balance_rows or []:
        if _alias_match(str(row.get("variable", "")), aliases):
            return row.get("smd_before"), row.get("smd_after")
    return None, None


def confounder_ledger(
    spec, dataset: str, *,
    covariates_used,
    balance_rows=None,
    overlap=None,
    e_value_point=None,
    adjustment: str = "",
    confounders=None,
    measure_fn=None,
) -> ConfounderLedger:
    """Build the adjustability ledger for one (trial, dataset).

    covariates_used: the model covariate column names (TTEResult.extra['covariates_used']).
    balance_rows:    [{variable, smd_before, smd_after}] (TTEResult.extra['balance']).
    overlap:         engine ps_overlap dict (TTEResult.extra['ps_overlap']).
    """
    measure_fn = measure_fn or _default_measure_fn
    confounders = confounders or STANDARD_ICU_CONFOUNDERS
    used = [str(c).lower().replace("_", " ") for c in (covariates_used or [])]

    rows: list[ConfounderRow] = []
    for concept, et, aliases in confounders:
        status, reason = measure_fn(concept, et, dataset)
        in_model = any(_alias_match(c, aliases) for c in used)
        if status in ("unmeasurable", "proxy"):
            cls = Adjustability.NOT_ADJUSTABLE          # residual confounding here
        elif in_model:
            cls = Adjustability.ADJUSTED
        else:
            cls = Adjustability.MEASURABLE_NOT_USED
        smd_b, smd_a = _smd_for(aliases, balance_rows)
        rows.append(ConfounderRow(
            confounder=concept, event_type=et.value, status=status, classification=cls,
            in_model=in_model, smd_before=smd_b, smd_after=smd_a, reason=reason))

    n = len(rows)
    n_adj = sum(r.classification == Adjustability.ADJUSTED for r in rows)
    n_mnu = sum(r.classification == Adjustability.MEASURABLE_NOT_USED for r in rows)
    n_na = sum(r.classification == Adjustability.NOT_ADJUSTABLE for r in rows)
    not_adj_names = [r.confounder for r in rows if r.classification == Adjustability.NOT_ADJUSTABLE]

    if not_adj_names:
        note = (f"{n_na} considered confounder(s) not adjustable in {dataset} "
                f"({', '.join(not_adj_names)}) -> residual confounding.")
        if e_value_point:
            note += (f" E-value {e_value_point:.2f}: an unmeasured confounder would need at "
                     f"least this association with both treatment and outcome to explain the effect.")
    else:
        note = f"All {n} considered confounders adjustable in {dataset}."

    summary = f"adjusted {n_adj}/{n} confounders"
    if n_na:
        summary += f"; {n_na} not-adjustable ({', '.join(not_adj_names)}) -> residual confounding"
        if e_value_point:
            summary += f" (E-value {e_value_point:.2f})"
    if overlap and overlap.get("poor"):
        summary += "; limited PS overlap (positivity concern)"

    return ConfounderLedger(
        nct_id=spec.nct_id, dataset=dataset, adjustment=adjustment, considered=rows,
        n_considered=n, n_adjusted=n_adj, n_measurable_not_used=n_mnu, n_not_adjustable=n_na,
        ps_overlap=overlap, e_value_point=e_value_point,
        residual_confounding_note=note, summary_line=summary,
    )


def ledger_from_comparison(comp, spec, *, measure_fn=None, confounders=None) -> ConfounderLedger:
    """Build the ledger straight from a ComparisonResult — reads covariates_used /
    balance / ps_overlap / e_value from the emulated TTEResult.extra."""
    ex = comp.emulated.extra or {}
    return confounder_ledger(
        spec, comp.dataset,
        covariates_used=ex.get("covariates_used") or [],
        balance_rows=ex.get("balance") or [],
        overlap=ex.get("ps_overlap"),
        e_value_point=ex.get("e_value_point"),
        adjustment=comp.emulated.method or "",
        confounders=confounders, measure_fn=measure_fn,
    )


def build_ledger_corpus(comparisons, specs, *, measure_fn=None, confounders=None) -> list[ConfounderLedger]:
    """One ConfounderLedger per comparison, joined to its spec by nct_id."""
    by_nct = {sp.nct_id: sp for sp in specs}
    out: list[ConfounderLedger] = []
    for c in comparisons:
        spec = by_nct.get(c.nct_id)
        if spec is None:
            continue
        out.append(ledger_from_comparison(c, spec, measure_fn=measure_fn, confounders=confounders))
    return out


def write_ledger_sidecar(comparisons, specs, path, **kwargs) -> int:
    """Build the ledger corpus and persist it as ledger.jsonl next to corpus.jsonl,
    joined on (nct_id, dataset). Returns the count written."""
    return dump_ledger_jsonl(build_ledger_corpus(comparisons, specs, **kwargs), path)


__all__ = [
    "STANDARD_ICU_CONFOUNDERS",
    "confounder_ledger",
    "ledger_from_comparison",
    "build_ledger_corpus",
    "write_ledger_sidecar",
]
