"""Calibration figure: emulated-vs-observed reliability plot (#60).

Renders #41's CalibrationCurve — it does NOT compute calibration (that's #41).
The curve (duck-typed, so figures stays import-light / no analysis import) carries:
  .points    : items with .emulated (raw OR/HR/RR), .observed (raw RR), .in_ci (bool)
               (dicts are also accepted)
  .slope, .intercept : OLS on the LOG scale -> the line is y = exp(intercept) * x**slope
                       (a straight line on log-log axes)
  .coverage  : fraction of points whose emulated CI covers the observed effect
  .n         : number of points

`calibration_points(curve)` is the PURE, import-light plot-ready extract (the #49
UI renders it without matplotlib); `calibration_plot(...)` is the journal figure
(lazy matplotlib, `viz` extra).
"""

from __future__ import annotations

import math


def _get(point, key):
    """Attr- or dict-style access, so a pydantic CalibrationPoint or a plain dict
    both work."""
    if isinstance(point, dict):
        return point.get(key)
    return getattr(point, key, None)


def calibration_points(curve) -> list[dict]:
    """Plot-ready (emulated, observed, in_ci) triples — pure, import-light."""
    out = []
    for p in getattr(curve, "points", []) or []:
        e, o = _get(p, "emulated"), _get(p, "observed")
        if e is None or o is None or e != e or o != o:  # skip missing/NaN
            continue
        out.append({"emulated": float(e), "observed": float(o),
                    "in_ci": bool(_get(p, "in_ci"))})
    return out


def calibration_plot(curve, path, *, title: str = "Emulated vs observed calibration") -> str:
    """Log-log scatter of emulated vs observed with the identity line and #41's
    fitted calibration line; points colored by whether the emulated CI covers the
    observed effect; annotated with slope + coverage. Returns the path. Needs the
    `viz` extra (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = calibration_points(curve)
    if not pts:
        raise ValueError("no finite (emulated, observed) points to plot")

    xs = [p["emulated"] for p in pts]
    ys = [p["observed"] for p in pts]
    lo = min(min(xs), min(ys)) * 0.8
    hi = max(max(xs), max(ys)) * 1.25

    fig, ax = plt.subplots(figsize=(5.2, 5))
    # identity y = x (perfect calibration)
    ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1, label="identity (y=x)")
    # #41's fitted log-log line: y = exp(intercept) * x**slope
    slope, intercept = getattr(curve, "slope", None), getattr(curve, "intercept", None)
    if slope is not None and intercept is not None:
        import numpy as np
        gx = np.geomspace(lo, hi, 50)
        ax.plot(gx, math.exp(intercept) * gx ** slope, color="#4575b4", lw=1.5,
                label=f"fit (slope={slope:.2f})")
    covered = [(x, y) for (x, y, p) in zip(xs, ys, pts) if p["in_ci"]]
    missed = [(x, y) for (x, y, p) in zip(xs, ys, pts) if not p["in_ci"]]
    if covered:
        ax.scatter(*zip(*covered), s=18, color="#1a9850", label="CI covers observed")
    if missed:
        ax.scatter(*zip(*missed), s=18, color="#d73027", label="CI misses observed")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("emulated effect"); ax.set_ylabel("observed effect")
    coverage = getattr(curve, "coverage", None)
    cov_txt = f"  coverage={coverage:.0%}" if isinstance(coverage, (int, float)) else ""
    ax.set_title(f"{title}{cov_txt}")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
