"""Confounder transparency for the per-trial detail (#104 sibling).

Renders probe's #105 adjustability artifacts (does NOT compute them):
- BALANCE (love plot): result.emulated.extra["balance"] = [{variable, smd_before, smd_after}].
- PS OVERLAP: result.emulated.extra["ps_overlap"] = {bin_centers, treated_density,
  control_density, frac_treated_off_support, overlap_coef, poor} | None.
- CONFOUNDER LEDGER: probe's ConfounderLedger sidecar (ledger.jsonl, joined by
  (nct_id, dataset)) — {summary_line, n_considered, n_adjusted, n_not_adjustable,
  e_value_point, residual_confounding_note, considered: [{confounder, status,
  classification, in_model, smd_before, smd_after, reason}]}.
  classification (our colors): adjusted=GREEN, measurable_not_used=AMBER,
  not_adjustable=RED (residual confounding).

PURE / import-light + duck-typed (dict OR pydantic), so the UI renders it with no
analysis/matplotlib and works before #105's loader is importable.
"""

from __future__ import annotations

_CLASS_COLOR = {           # green / amber / red
    "adjusted": "#3f8f86",
    "measurable_not_used": "#b8823c",
    "not_adjustable": "#c0392b",
}


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def balance_rows(extra) -> list[dict]:
    """Normalized SMD-before/after rows from the engine's balance table."""
    out = []
    for b in (extra or {}).get("balance") or []:
        out.append({"variable": _get(b, "variable"),
                    "smd_before": _get(b, "smd_before"), "smd_after": _get(b, "smd_after")})
    return out


def ledger_rows(ledger) -> list[dict]:
    """Render rows from a ConfounderLedger's `considered` list, colored by
    classification. Empty when no ledger."""
    rows = []
    for r in (_get(ledger, "considered") or []):
        cls = _get(r, "classification") or "adjusted"
        rows.append({
            "name": _get(r, "confounder"), "classification": cls,
            "status": _get(r, "status"), "in_model": _get(r, "in_model"),
            "smd_before": _get(r, "smd_before"), "smd_after": _get(r, "smd_after"),
            "reason": _get(r, "reason"), "color": _CLASS_COLOR.get(cls, "#8b8598"),
        })
    return rows


def _fallback_ledger_rows(extra) -> list[dict]:
    """No ledger sidecar: the only thing we know is the ADJUSTED set (green) from
    the balance table."""
    return [{"name": r["variable"], "classification": "adjusted", "status": "measurable",
             "in_model": True, "smd_before": r["smd_before"], "smd_after": r["smd_after"],
             "reason": None, "color": _CLASS_COLOR["adjusted"]}
            for r in balance_rows(extra)]


def confounder_summary(extra, *, ledger=None, threshold: float = 0.1) -> dict:
    """Counts + the one-line summary for the per-trial header + all-trials column."""
    if ledger is not None:
        n_adj, n_con = _get(ledger, "n_adjusted"), _get(ledger, "n_considered")
        line = _get(ledger, "summary_line") or (f"adjusted {n_adj}/{n_con}" if n_con else "—")
        return {
            "label": line, "n_adjusted": n_adj, "n_considered": n_con,
            "n_not_adjustable": _get(ledger, "n_not_adjustable"),
            "n_measurable_not_used": _get(ledger, "n_measurable_not_used"),
            "e_value": _get(ledger, "e_value_point"),
            "residual_note": _get(ledger, "residual_confounding_note"),
        }
    rows = balance_rows(extra)
    n = len(rows)
    return {"label": f"adjusted {n}/{n}" if n else "—", "n_adjusted": n, "n_considered": n,
            "n_not_adjustable": 0, "n_measurable_not_used": 0, "e_value": None,
            "residual_note": None}


def confounder_block(extra, *, ledger=None) -> dict | None:
    """Full per-trial confounder block (ledger rows + balance + ps_overlap + summary)
    for a card; None when there's nothing to show."""
    bal = balance_rows(extra)
    ps = _get(ledger, "ps_overlap") or (extra or {}).get("ps_overlap")
    rows = ledger_rows(ledger) if ledger is not None else _fallback_ledger_rows(extra)
    if not bal and not rows and not ps:
        return None
    return {
        "ledger": rows,
        "balance": bal,
        "ps_overlap": ps,
        "summary": confounder_summary(extra, ledger=ledger),
    }
