"""Confounder transparency for the per-trial detail (#104 sibling).

Makes confounding VISIBLE: a ledger (adjusted / measurable-not-used / not-adjustable),
a balance (love-plot) view of SMD before vs after adjustment, and a propensity-score
overlap view. PURE / import-light (reads the persisted ComparisonResult.extra +
worker1/probe's ledger when present) so the UI renders it without analysis/matplotlib.

Data sources:
- balance: result.emulated.extra["balance"] = [{variable, smd_before, smd_after}] (#10 engine).
- ledger:  probe exposes [{name, status}] where status in
           {"adjusted","measurable_unused","unmeasured"}; absent -> derive the
           ADJUSTED set (green) from the balance table.
- ps_overlap: probe exposes extra["ps_overlap"] (density bins) — passed through.
"""

from __future__ import annotations

_STATUS_COLOR = {           # green / amber / red
    "adjusted": "#3f8f86",
    "measurable_unused": "#b8823c",
    "unmeasured": "#c0392b",
    "not_adjustable": "#c0392b",
}


def balance_rows(extra) -> list[dict]:
    """Normalized SMD-before/after rows from the engine's balance table."""
    out = []
    for b in (extra or {}).get("balance") or []:
        d = b if isinstance(b, dict) else getattr(b, "__dict__", {})
        out.append({"variable": d.get("variable"),
                    "smd_before": d.get("smd_before"), "smd_after": d.get("smd_after")})
    return out


def confounder_ledger(extra, *, ledger=None) -> list[dict]:
    """Per-confounder status + SMDs. Uses probe's ledger when supplied; otherwise
    falls back to the adjusted set (green) inferred from the balance table."""
    bal = {r["variable"]: r for r in balance_rows(extra)}
    if ledger:
        rows = []
        for item in ledger:
            d = item if isinstance(item, dict) else {"name": getattr(item, "name", None),
                                                     "status": getattr(item, "status", None)}
            b = bal.get(d.get("name"), {})
            rows.append({"name": d.get("name"), "status": d.get("status") or "adjusted",
                         "smd_before": b.get("smd_before"), "smd_after": b.get("smd_after"),
                         "color": _STATUS_COLOR.get(d.get("status") or "adjusted", "#8b8598")})
        return rows
    return [{"name": v, "status": "adjusted", "smd_before": r.get("smd_before"),
             "smd_after": r.get("smd_after"), "color": _STATUS_COLOR["adjusted"]}
            for v, r in bal.items()]


def confounder_summary(extra, *, ledger=None, threshold: float = 0.1) -> dict:
    """Counts for the compact per-trial summary + the all-trials table column."""
    led = confounder_ledger(extra, ledger=ledger)
    total = len(led)
    n_adjusted = sum(1 for r in led if r["status"] == "adjusted")
    after = [abs(r["smd_after"]) for r in led if r.get("smd_after") is not None]
    return {
        "n_adjusted": n_adjusted,
        "n_total": total,
        "n_unmeasured": sum(1 for r in led if r["status"] in ("unmeasured", "not_adjustable")),
        "n_measurable_unused": sum(1 for r in led if r["status"] == "measurable_unused"),
        "n_balanced_after": sum(1 for s in after if s <= threshold),
        "max_smd_after": max(after, default=None),
        "label": f"adjusted {n_adjusted}/{total}" if total else "—",
    }


def confounder_block(extra, *, ledger=None) -> dict | None:
    """Full per-trial confounder block (ledger + balance + ps_overlap + summary) for
    a card; None when there's nothing to show (keeps cards clean for synthetic runs)."""
    bal = balance_rows(extra)
    ps = (extra or {}).get("ps_overlap")
    if not bal and not ledger and not ps:
        return None
    return {
        "ledger": confounder_ledger(extra, ledger=ledger),
        "balance": bal,
        "ps_overlap": ps,
        "summary": confounder_summary(extra, ledger=ledger),
    }
