"""Kaplan-Meier survival figure (#60).

Unlike the forest/calibration figures (which read the summary corpus), KM needs
PER-PATIENT survival — a cohort frame with a follow-up time + 0/1 event per
trajectory, by arm. So km_data/km_plot take such a frame (e.g. a future
time-to-event build_analysis_frame), NOT the persisted ComparisonResult corpus.

`km_data` computes the KM estimator in PURE python (no lifelines/numpy — figures
stays light); `km_plot` renders step curves via matplotlib (lazy, `viz` extra).
The pure curve also feeds the #49 UI without matplotlib.
"""

from __future__ import annotations

from itertools import groupby


def km_survival(times, events) -> list[tuple[float, float]]:
    """Kaplan-Meier survival curve from follow-up `times` + 0/1 `events`
    (1=event, 0=censored). Returns step points [(t, S(t)), ...] starting at (0,1).
    Pure python; product-limit estimator."""
    pairs = sorted((float(t), int(e)) for t, e in zip(times, events))
    at_risk = len(pairs)
    if at_risk == 0:
        return [(0.0, 1.0)]
    surv = 1.0
    curve = [(0.0, 1.0)]
    for t, grp in groupby(pairs, key=lambda x: x[0]):
        grp = list(grp)
        deaths = sum(e for _, e in grp)
        if deaths and at_risk:
            surv *= 1.0 - deaths / at_risk
            curve.append((t, surv))
        at_risk -= len(grp)
    return curve


def km_data(frame, *, time_col: str = "time", event_col: str = "event",
            group_col: str = "group") -> dict[str, list[tuple[float, float]]]:
    """Per-group KM curves from a cohort survival frame. Pure / import-light
    (column access only — the caller supplies the DataFrame). For the #49 UI."""
    groups = list(dict.fromkeys(frame[group_col].tolist()))
    out: dict[str, list[tuple[float, float]]] = {}
    for g in groups:
        mask = frame[group_col] == g
        out[str(g)] = km_survival(frame.loc[mask, time_col].tolist(),
                                  frame.loc[mask, event_col].tolist())
    return out


def km_plot(frame, path, *, time_col: str = "time", event_col: str = "event",
            group_col: str = "group", title: str = "Kaplan-Meier survival") -> str:
    """Step KM curves per arm to `path`. Needs the `viz` extra (matplotlib)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = km_data(frame, time_col=time_col, event_col=event_col, group_col=group_col)
    if not curves:
        raise ValueError("no groups to plot")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for g, curve in curves.items():
        ts = [t for t, _ in curve]
        ss = [s for _, s in curve]
        ax.step(ts, ss, where="post", label=f"{g} (n={int((frame[group_col] == g).sum())})")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time"); ax.set_ylabel("survival S(t)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
