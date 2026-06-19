"""
Generalized target-trial-emulation (TTE) analysis engine.

Operates on an *analysis-ready* cohort DataFrame: one row per ICU stay, with a
binary treatment column, a set of numeric covariates, and one or more outcomes.
Each outcome is either:
  - survival: (time_col, event_col)  -> Cox PH hazard ratio / log-rank
  - binary:   (event_col)            -> logistic odds ratio / risk diff / 2x2 tests

The engine supports configurable confounding adjustment (propensity-score
matching, IPTW, covariate-adjusted regression, or unadjusted) and a choice of
primary statistical test per outcome. All functions are pure and fast so the
webapp can re-run them interactively as the user tunes the design.

Refactored & generalized from mimic_demo/06_survival_analysis.py.

Ported into tteEngine for #10 (probe) as a SELF-CONTAINED copy of trialsim's
app/trialsim/tte_engine.py (per jpic's directive: no cross-repo imports). The
typed entrypoint + TTEResult live in runner.py; the heavy estimators (PSM/IPTW/
Cox/AIPW/KM/E-values) stay here, behind the `analysis` optional-deps extra
(lifelines / statsmodels / scikit-learn).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd

# Silence only the benign convergence/specification chatter the modelling libraries
# emit on hard cohorts — NOT a blanket ignore, so genuinely informative warnings
# (pandas SettingWithCopy, numpy errors, deprecations) still surface. Categories are
# imported defensively: a library version that renames/moves a class must not break
# the import (we just skip that filter).
def _quiet_modelling_warnings():
    specs = [
        ("sklearn.exceptions", ["ConvergenceWarning"]),
        ("statsmodels.tools.sm_exceptions",
         ["ConvergenceWarning", "IterationLimitWarning", "HessianInversionWarning",
          "PerfectSeparationWarning", "SpecificationWarning", "ValueWarning"]),
        ("lifelines.exceptions",
         ["ConvergenceWarning", "ApproximationWarning", "StatisticalWarning"]),
    ]
    import importlib
    for mod, names in specs:
        try:
            m = importlib.import_module(mod)
        except Exception:
            continue
        for nm in names:
            cat = getattr(m, nm, None)
            if isinstance(cat, type) and issubclass(cat, Warning):
                warnings.filterwarnings("ignore", category=cat)


_quiet_modelling_warnings()

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance as scipy_distance
from scipy import stats as scipy_stats
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import statsmodels.api as sm

RS = 42
SMD_THRESHOLD = 0.1


def interpret_result(r: dict) -> dict:
    """Combine the diagnostics of one result into a plain-language robustness
    verdict. Returns {level, badge, caption, reasons}. Heuristic, transparent —
    meant to orient the user, not replace judgement."""
    if not r or not r.get("ok"):
        return {"level": "n/a", "badge": "—", "caption": "Not estimable.", "reasons": []}

    p = r.get("p_value", 1.0)
    est = r.get("estimate", float("nan"))
    name = r.get("estimate_name", "")
    is_ratio = name in ("Hazard Ratio", "Odds Ratio")
    ev = (r.get("e_value") or {}).get("point")
    ev_ci = (r.get("e_value") or {}).get("ci")
    poor_overlap = bool((r.get("overlap") or {}).get("poor"))
    ph_violated = bool(r.get("ph_violated"))
    sig = p is not None and p < 0.05
    reasons = []

    # large effect on the ratio scale flags possible residual/indication confounding
    extreme = is_ratio and np.isfinite(est) and (est >= 1.5 or est <= 0.667)

    if not sig:
        level, badge = "inconclusive", "⚪"
        reasons.append("confidence interval crosses the null (not significant)")
    elif poor_overlap:
        level, badge = "fragile", "🟠"
        reasons.append("limited propensity-score overlap (positivity concern)")
    elif ev_ci is not None and ev_ci < 1.25:
        level, badge = "fragile", "🟠"
        reasons.append(f"weak E-value for the CI ({ev_ci:.2f}) — modest confounding could explain it")
    elif extreme:
        level, badge = "significant — confounding caution", "🟡"
        reasons.append("large effect size: if treatment is escalation/rescue therapy, "
                       "confounding by indication may persist despite adjustment")
        if ev:
            reasons.append(f"E-value {ev:.2f} (a confounder this strong would be needed to nullify it)")
    else:
        level, badge = "robust", "🟢"
        reasons.append(f"significant, adequate overlap" + (f", E-value {ev:.2f}" if ev else ""))
    if ph_violated:
        reasons.append("proportional-hazards assumption violated — HR is a time-averaged effect")

    cap = f"{badge} {level.capitalize()} — " + "; ".join(reasons) + "."
    return {"level": level, "badge": badge, "caption": cap, "reasons": reasons}


def replication_summary(results: list) -> list:
    """For each outcome estimated on ≥2 datasets (All-patients), report whether
    the datasets agree in direction and significance — an external-replication
    check across MIMIC and eICU."""
    by_outcome = {}
    for r in results:
        if r.get("ok") and "All" in r.get("subgroup", ""):
            by_outcome.setdefault(r["outcome_key"], []).append(r)
    out = []
    for ok_key, rs in by_outcome.items():
        if len(rs) < 2:
            continue
        name = rs[0].get("estimate_name", "")
        null = 0.0 if name == "Risk Difference" else 1.0
        dirs = {("up" if r["estimate"] > null else "down") for r in rs}
        sigs = [r["p_value"] < 0.05 for r in rs]
        if len(dirs) == 1 and all(sigs):
            verdict, badge = "replicated (same direction, both significant)", "🟢"
        elif len(dirs) == 1:
            verdict, badge = "consistent direction (not significant in all)", "🟡"
        else:
            verdict, badge = "discordant across datasets", "🟠"
        out.append({"outcome": rs[0]["outcome_label"], "badge": badge, "verdict": verdict,
                    "estimates": {r["dataset"]: round(r["estimate"], 2) for r in rs}})
    return out


def e_value(estimate, ci_low, ci_high):
    """E-value (VanderWeele & Ding 2017) for a ratio estimand (HR/OR/RR):
    the minimum strength of an unmeasured confounder, on the risk-ratio scale,
    needed to explain away the observed association and its CI bound nearest the
    null. For HR/OR this is the standard approximation that treats the ratio as a
    risk ratio. Returns {'point', 'ci'} or None if not computable."""
    vals = [estimate, ci_low, ci_high]
    if any(v is None or not np.isfinite(v) or v <= 0 for v in vals):
        return None

    def _ev(x):
        x = x if x >= 1 else 1.0 / x
        return x + np.sqrt(x * (x - 1.0))

    e_point = _ev(estimate)
    if estimate >= 1:
        crosses, limit = ci_low <= 1.0, ci_low
    else:
        crosses, limit = ci_high >= 1.0, ci_high
    e_ci = 1.0 if crosses else _ev(limit)
    return {"point": float(e_point), "ci": float(e_ci)}


# ──────────────────────────────────────────────────────────────────────────
# Specs
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class OutcomeSpec:
    key: str
    label: str
    kind: str            # 'survival' | 'binary'
    event_col: str
    time_col: Optional[str] = None
    reverse: bool = False        # HR<1 = benefit when reversed (e.g. time-to-discharge)
    restrict_col: Optional[str] = None   # only analyse rows where restrict_col == 1 (e.g. baseline_mv)
    horizon_days: float = 28.0
    benefit_direction: str = "lower"     # 'lower' = lower effect estimate is better


@dataclass
class AnalysisConfig:
    treatment_col: str = "T"
    covariates: list = field(default_factory=list)
    adjustment: str = "psm"      # 'psm' | 'iptw' | 'covariate' | 'unadjusted'
    match_ratio: int = 4
    caliper_mad: float = 1.0
    survival_test: str = "cox"   # 'cox' | 'logrank'
    binary_test: str = "logistic"  # 'logistic' | 'chi2' | 'fisher' | 'riskdiff'
    iptw_stabilized: bool = True
    iptw_trim: float = 0.01      # symmetric PS trimming for IPTW stability
    id_col: str = "patientunitstayid"   # cluster unit for robust SEs under matching


# ──────────────────────────────────────────────────────────────────────────
# Balance diagnostics
# ──────────────────────────────────────────────────────────────────────────
def _mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def baseline_table(data: pd.DataFrame, covars, treat_col="T", pretty=None):
    """Trial 'Table 1': per-covariate mean (SD) by arm plus SMD. Returns a list of
    row dicts suitable for a dataframe."""
    smd = compute_smd(data, covars, treat_col)
    t1 = data[data[treat_col] == 1]
    t0 = data[data[treat_col] == 0]
    rows = []
    for c in covars:
        x1 = pd.to_numeric(t1[c], errors="coerce")
        x0 = pd.to_numeric(t0[c], errors="coerce")
        # cohort columns carry the "_day1" suffix (not "_day_1"); strip it before
        # prettifying so Table 1 reads "creatinine", not "creatinine day1".
        name = (pretty or {}).get(c, c).replace("_day1", "").replace("_", " ").strip()
        rows.append({
            "Covariate": name,
            "Treated (mean±SD)": f"{x1.mean():.1f} ± {x1.std():.1f}",
            "Control (mean±SD)": f"{x0.mean():.1f} ± {x0.std():.1f}",
            "SMD": round(float(smd.get(c, np.nan)), 3),
        })
    rows.sort(key=lambda r: -(r["SMD"] if r["SMD"] == r["SMD"] else 0))
    return rows


def compute_smd(data: pd.DataFrame, covars, treat_col="T", weights=None) -> pd.Series:
    """Standardized mean difference per covariate (optionally weighted)."""
    t = data[treat_col].values
    out = {}
    for c in covars:
        x = pd.to_numeric(data[c], errors="coerce").values
        m = ~np.isnan(x)
        x1, x0 = x[m & (t == 1)], x[m & (t == 0)]
        if weights is not None:
            w = np.asarray(weights)[m]
            w1, w0 = w[(t[m] == 1)], w[(t[m] == 0)]
            mu1 = np.average(x1, weights=w1) if len(x1) else np.nan
            mu0 = np.average(x0, weights=w0) if len(x0) else np.nan
            v1 = np.average((x1 - mu1) ** 2, weights=w1) if len(x1) else np.nan
            v0 = np.average((x0 - mu0) ** 2, weights=w0) if len(x0) else np.nan
        else:
            mu1, mu0 = (x1.mean() if len(x1) else np.nan), (x0.mean() if len(x0) else np.nan)
            v1, v0 = (x1.var() if len(x1) else np.nan), (x0.var() if len(x0) else np.nan)
        pooled = np.sqrt((v1 + v0) / 2)
        out[c] = abs(mu1 - mu0) / pooled if pooled and pooled > 0 else 0.0
    return pd.Series(out).fillna(0.0)


# ──────────────────────────────────────────────────────────────────────────
# Propensity scores & adjustment
# ──────────────────────────────────────────────────────────────────────────
def propensity_scores(df: pd.DataFrame, covars, treat_col="T", C=1.0) -> np.ndarray:
    X = df[covars].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())
    # Standardize covariates: ICU covariates span wildly different scales (creatinine
    # ~0–25 vs platelets ~0–2000), which (a) makes lbfgs fail to converge and (b) makes
    # the L2 penalty scale-dependent — larger-scale covariates get effectively less
    # regularization, distorting the PS model. StandardScaler fixes both (zero-variance
    # columns are handled by sklearn; the engine also drops constant covariates upstream).
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000, random_state=RS, C=C)
    lr.fit(Xs, df[treat_col])
    return lr.predict_proba(Xs)[:, 1]


def propensity_match_1toN(df_in, covars, treat_col="T", N=4, caliper_mad=1.0):
    """1:N nearest-neighbour PS matching WITH replacement (Mahalanobis≡|ΔPS| in 1D)."""
    df_work = df_in.copy().reset_index(drop=True)
    df_work["ps"] = propensity_scores(df_work, covars, treat_col)

    treated = df_work[df_work[treat_col] == 1]
    control = df_work[df_work[treat_col] == 0]
    t_idx, c_idx = treated.index.tolist(), control.index.tolist()
    if not t_idx or not c_idx:
        return df_work.iloc[0:0].copy(), df_work

    dists = scipy_distance.cdist(
        treated["ps"].values.reshape(-1, 1),
        control["ps"].values.reshape(-1, 1),
        metric="euclidean",
    )
    caliper = caliper_mad * _mad(dists.flatten())

    rows = []
    for i in range(len(t_idx)):
        used, matches = set(), []
        for j in np.argsort(dists[i, :]):
            if dists[i, j] > caliper:
                break
            cj = c_idx[j]
            if cj not in used:
                matches.append(cj)
                used.add(cj)
            if len(matches) >= N:
                break
        # Only keep a treated unit that found at least one control within the caliper.
        # Retaining unmatched treated (no in-caliper control) would put them in the
        # treated arm with no comparable control — reintroducing the positivity
        # violation the caliper is meant to enforce and biasing the matched estimate.
        if matches:
            rows.append(t_idx[i])
            rows.extend(matches)
    matched = df_work.loc[rows].copy()
    return matched, df_work


def ps_overlap(df, covars, treat_col="T", bins=20):
    """Propensity-score overlap (positivity) diagnostic between arms.
    Returns histogram data plus the fraction of treated patients whose PS lies
    outside the control PS range (poor common support)."""
    ps = propensity_scores(df, covars, treat_col)
    t = df[treat_col].values
    ps_t, ps_c = ps[t == 1], ps[t == 0]
    if len(ps_t) == 0 or len(ps_c) == 0:
        return None
    edges = np.linspace(0, 1, bins + 1)
    h_t, _ = np.histogram(ps_t, bins=edges, density=True)
    h_c, _ = np.histogram(ps_c, bins=edges, density=True)
    c_lo, c_hi = ps_c.min(), ps_c.max()
    off = float(np.mean((ps_t < c_lo) | (ps_t > c_hi)))
    # overlap coefficient of the two normalized histograms (0=disjoint,1=identical)
    pt = h_t / (h_t.sum() or 1)
    pc = h_c / (h_c.sum() or 1)
    overlap_coef = float(np.minimum(pt, pc).sum())
    return {
        "bin_centers": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "treated_density": h_t.tolist(), "control_density": h_c.tolist(),
        "frac_treated_off_support": off, "overlap_coef": overlap_coef,
        "poor": bool(off > 0.10 or overlap_coef < 0.5),
    }


def aipw_riskdiff(df, spec: "OutcomeSpec", covars, treat_col="T", trim=0.01):
    """Augmented IPW (doubly-robust) ATE risk difference for a binary outcome.

    Combines a propensity model e(X) with outcome models m₁(X)=E[Y|X,T=1] and
    m₀(X)=E[Y|X,T=0]. The estimator is CONSISTENT if EITHER model is correctly
    specified (double robustness) — more robust than IPTW or outcome-regression alone.
    SE is the empirical influence-function SD / √n (sandwich-equivalent), so it's honest
    without bootstrapping. Returns the same dict shape as run_riskdiff, or None/{error}.
    """
    covs = [c for c in covars if c in df.columns]
    y = pd.to_numeric(df[spec.event_col], errors="coerce")
    keep = pd.concat([y.rename("_y"), df[[treat_col] + covs].apply(pd.to_numeric, errors="coerce")],
                     axis=1).dropna()
    if keep[treat_col].nunique() < 2 or keep["_y"].nunique() < 2 or len(keep) < 40:
        return None
    covs = [c for c in covs if keep[c].nunique() > 1]
    t = keep[treat_col].values.astype(float)
    yv = keep["_y"].values.astype(float)
    # propensity e(X), trimmed for positivity
    e = np.clip(propensity_scores(keep, covs, treat_col) if covs else np.full(len(keep), t.mean()),
                trim, 1 - trim)
    # outcome models: one logistic on covariates within each arm (fall back to arm mean)
    def _outcome_pred(arm):
        g = keep[keep[treat_col] == arm]
        if not covs or g["_y"].nunique() < 2:
            return np.full(len(keep), g["_y"].mean())
        sc = StandardScaler().fit(keep[covs])          # same scaling for fit and predict
        gm = LogisticRegression(max_iter=2000, random_state=RS)
        gm.fit(sc.transform(g[covs]), g["_y"])
        return gm.predict_proba(sc.transform(keep[covs]))[:, 1]
    m1, m0 = _outcome_pred(1), _outcome_pred(0)
    # AIPW influence function for the ATE (risk difference)
    psi1 = t * (yv - m1) / e + m1
    psi0 = (1 - t) * (yv - m0) / (1 - e) + m0
    tau = psi1 - psi0
    rd = float(np.mean(tau))
    se = float(np.std(tau, ddof=1) / np.sqrt(len(tau)))
    lo, hi = rd - 1.96 * se, rd + 1.96 * se
    p = float(2 * scipy_stats.norm.sf(abs(rd / se))) if se > 0 else np.nan
    et = float(np.average(yv[t == 1])) if (t == 1).any() else np.nan
    ec = float(np.average(yv[t == 0])) if (t == 0).any() else np.nan
    return {
        "estimate_name": "Risk Difference", "estimate": rd, "ci_low": float(lo),
        "ci_high": float(hi), "p_value": p, "n": int(len(keep)),
        "n_treated": int((t == 1).sum()), "n_control": int((t == 0).sum()),
        "event_treated": float(np.mean(psi1)), "event_control": float(np.mean(psi0)),
        "risk_difference": rd, "rd_ci_low": float(lo), "rd_ci_high": float(hi),
        "test": "Doubly-robust ATE (AIPW)", "e_value": None,
    }


def iptw_weights(df, covars, treat_col="T", stabilized=True, trim=0.01):
    ps = propensity_scores(df, covars, treat_col)
    ps = np.clip(ps, trim, 1 - trim)
    t = df[treat_col].values
    if stabilized:
        p_treat = t.mean()
        w = np.where(t == 1, p_treat / ps, (1 - p_treat) / (1 - ps))
    else:
        w = np.where(t == 1, 1.0 / ps, 1.0 / (1 - ps))
    return w, ps


# ──────────────────────────────────────────────────────────────────────────
# Effect estimation
# ──────────────────────────────────────────────────────────────────────────
def _prep_survival(data, spec: OutcomeSpec, treat_col, extra_cols=None):
    cols = [spec.time_col, spec.event_col, treat_col] + (extra_cols or [])
    sub = data[[c for c in cols if c in data.columns]].dropna(subset=[spec.time_col, spec.event_col, treat_col])
    sub = sub[sub[spec.time_col] > 0].copy()
    return sub


def _arm_event_rates(sub, treat_col, event_col, wcol=None):
    """(treated, control) event rate. Weighted by `wcol` when present (IPTW pseudo-
    population) so the absolute-risk difference stays consistent with the weighted HR."""
    def rate(arm):
        g = sub[sub[treat_col] == arm]
        if not len(g):
            return float("nan")
        if wcol and g[wcol].sum() > 0:
            return float(np.average(g[event_col], weights=g[wcol]))
        return float(g[event_col].mean())
    return rate(1), rate(0)


def run_cox(data, spec: OutcomeSpec, treat_col="T", weights=None, adjust_covariates=None,
            compute_ph=True, cluster_col=None):
    adj = [c for c in (adjust_covariates or []) if c in data.columns and c != treat_col]
    clmem = cluster_col if (cluster_col and cluster_col in data.columns) else None
    # Attach IPTW weights as a column BEFORE row-subsetting so they stay aligned to
    # their patient when _prep_survival drops rows (NaN/non-positive times). Slicing
    # the weight array positionally after the drop would misassign weights.
    if weights is not None:
        data = data.copy()
        w_arr = np.asarray(weights)
        data["_w"] = w_arr if len(w_arr) == len(data) else np.resize(w_arr, len(data))
    extra = adj + ([clmem] if clmem else []) + (["_w"] if weights is not None else [])
    sub = _prep_survival(data, spec, treat_col, extra_cols=extra)
    if len(sub) < 20 or sub[treat_col].nunique() < 2:
        return None
    # drop covariates with no variance, then z-score and prune near-collinear
    adj = [c for c in adj if sub[c].nunique() > 1]
    if adj:
        sub = sub.copy()
        sub[adj] = (sub[adj] - sub[adj].mean()) / sub[adj].std(ddof=0).replace(0, np.nan)
        sub[adj] = sub[adj].fillna(0.0)
        corr = sub[adj].corr().abs()
        drop = set()
        for i, ci in enumerate(adj):
            for cj in adj[i + 1:]:
                if cj not in drop and corr.loc[ci, cj] > 0.9:
                    drop.add(cj)
        adj = [c for c in adj if c not in drop]
    # need at least a few events in each arm for a meaningful HR
    ev = sub.groupby(treat_col)[spec.event_col].sum()
    if ev.min() < 3:
        return {"error": "too few events in one arm to estimate a hazard ratio"}
    wcol = "_w" if weights is not None else None
    # cluster-robust SEs when matching reuses controls (cluster on patient id)
    use_cluster = clmem is not None and wcol is None and sub[clmem].duplicated().any()
    fit_cols = ([spec.time_col, spec.event_col, treat_col] + adj
                + ([clmem] if use_cluster else []) + ([wcol] if wcol else []))

    def _fit(pen):
        m = CoxPHFitter(penalizer=pen)
        m.fit(sub[fit_cols], duration_col=spec.time_col, event_col=spec.event_col,
              weights_col=wcol, cluster_col=clmem if use_cluster else None,
              robust=(weights is not None) or use_cluster)
        return m

    cph = None
    for pen in ([0.1] if adj else [0.0, 0.1, 1.0]):
        try:
            cph = _fit(pen)
            break
        except Exception:
            continue
    if cph is None:
        return {"error": "Cox model did not converge (degenerate risk sets)"}
    s = cph.summary
    hr, lo, hi = (float(np.exp(s.loc[treat_col, c])) for c in ("coef", "coef lower 95%", "coef upper 95%"))
    p = float(s.loc[treat_col, "p"])
    if spec.reverse:
        hr, lo, hi = 1 / hr, 1 / hi, 1 / lo

    # Proportional-hazards diagnostic (Schoenfeld) for the treatment term
    ph_p = None
    if compute_ph:
        try:
            if wcol is None:
                pht = proportional_hazard_test(cph, sub[fit_cols], time_transform="rank")
                ph_p = float(pht.summary.loc[treat_col, "p"])
        except Exception:
            ph_p = None

    # Arm event rates for the absolute-risk difference / NNT. Under IPTW the HR is
    # weight-adjusted, so the event rates must be too (the IPTW pseudo-population),
    # otherwise a crude RD/NNT would be inconsistent with the adjusted HR. PSM already
    # adjusts by matched-sample membership, so its (unweighted) rates are correct.
    et, ec = _arm_event_rates(sub, treat_col, spec.event_col, wcol)
    return {
        "estimate_name": "Hazard Ratio", "estimate": hr, "ci_low": lo, "ci_high": hi,
        "p_value": p, "n": int(len(sub)),
        "n_treated": int((sub[treat_col] == 1).sum()), "n_control": int((sub[treat_col] == 0).sum()),
        "event_treated": et, "event_control": ec,
        "test": "Cox proportional hazards" + (" (IPTW)" if weights is not None else "")
                + (" (cluster-robust)" if use_cluster else ""),
        "ph_pvalue": ph_p, "ph_violated": (ph_p is not None and ph_p < 0.05),
        "e_value": e_value(hr, lo, hi),
    }


def run_logrank(data, spec: OutcomeSpec, treat_col="T"):
    sub = _prep_survival(data, spec, treat_col)
    if len(sub) < 20 or sub[treat_col].nunique() < 2:
        return None
    g1 = sub[sub[treat_col] == 1]
    g0 = sub[sub[treat_col] == 0]
    res = logrank_test(g1[spec.time_col], g0[spec.time_col], g1[spec.event_col], g0[spec.event_col])
    # accompany with a Cox HR for an effect size (may fail; log-rank p still valid)
    cox = run_cox(data, spec, treat_col)
    if not cox or "estimate" not in cox:
        cox = None
    out = {
        "estimate_name": "Hazard Ratio", "estimate": cox["estimate"] if cox else np.nan,
        "ci_low": cox["ci_low"] if cox else np.nan, "ci_high": cox["ci_high"] if cox else np.nan,
        "p_value": float(res.p_value), "n": int(len(sub)),
        "n_treated": int(len(g1)), "n_control": int(len(g0)),
        "event_treated": float(g1[spec.event_col].mean()), "event_control": float(g0[spec.event_col].mean()),
        "test": "Log-rank test", "test_statistic": float(res.test_statistic),
    }
    return out


def run_logistic(data, spec: OutcomeSpec, treat_col="T", covariates=None, weights=None,
                 cluster_col=None):
    """Logistic OR for treatment; optionally covariate-adjusted, weighted, and/or
    cluster-robust (clustered on patient id when matching reuses controls)."""
    y = pd.to_numeric(data[spec.event_col], errors="coerce")
    X_cols = [treat_col] + (covariates or [])
    keep = [treat_col] + (covariates or [])
    clmem = cluster_col if (cluster_col and cluster_col in data.columns) else None
    frame = pd.concat([y.rename("_y"), data[keep].apply(pd.to_numeric, errors="coerce"),
                       (data[clmem].rename("_cl") if clmem else None)], axis=1).dropna()
    d = frame
    if len(d) < 20 or d[treat_col].nunique() < 2 or d["_y"].nunique() < 2:
        return None
    Xd = sm.add_constant(d[X_cols], has_constant="add")
    use_cluster = clmem is not None and d["_cl"].duplicated().any() and weights is None
    try:
        if weights is not None:
            w = pd.Series(np.asarray(weights), index=data.index).reindex(d.index).values
            # IPTW weights are NOT frequency counts: a model-based SE on a freq-weighted
            # GLM is anti-conservative (it acts as if N=Σw). Use a robust sandwich,
            # clustering each independent patient, for honest CIs — mirroring the
            # cluster-robust Cox IPTW path. (statsmodels emits a benign support warning.)
            grp = d["_cl"].values if clmem is not None else np.arange(len(d))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(d["_y"], Xd, family=sm.families.Binomial(),
                             freq_weights=w).fit(cov_type="cluster",
                                                 cov_kwds={"groups": grp})
        elif use_cluster:
            res = sm.Logit(d["_y"], Xd).fit(disp=0, cov_type="cluster",
                                            cov_kwds={"groups": d["_cl"].values})
        else:
            res = sm.Logit(d["_y"], Xd).fit(disp=0)
    except Exception as e:
        return {"error": str(e)}
    coef = res.params[treat_col]
    ci = res.conf_int().loc[treat_col]
    or_, lo, hi = np.exp(coef), np.exp(ci[0]), np.exp(ci[1])
    p = float(res.pvalues[treat_col])
    label = "Odds Ratio"
    # weighted arm event rates under IPTW (consistent with the weighted OR), crude otherwise
    wcol_d = None
    if weights is not None:
        d = d.assign(_w=pd.Series(np.asarray(weights), index=data.index).reindex(d.index).values)
        wcol_d = "_w"
    et, ec = _arm_event_rates(d, treat_col, "_y", wcol_d)
    return {
        "estimate_name": label, "estimate": float(or_), "ci_low": float(lo), "ci_high": float(hi),
        "p_value": p, "n": int(len(d)),
        "n_treated": int((d[treat_col] == 1).sum()), "n_control": int((d[treat_col] == 0).sum()),
        "event_treated": et, "event_control": ec,
        "test": "Logistic regression" + (" (adjusted)" if covariates else "")
                + (" (IPTW)" if weights is not None else "")
                + (" (cluster-robust)" if use_cluster else ""),
        "e_value": e_value(float(or_), float(lo), float(hi)),
    }


def run_riskdiff(data, spec: OutcomeSpec, treat_col="T", weights=None, cluster_col=None):
    """Risk difference (treated − control). Under IPTW (weights present) it is the
    WEIGHTED RD from a linear-probability GLM with a robust sandwich SE — so it matches
    the weighted HR/OR instead of being a crude 2×2 RD that silently ignores the weights.
    Unweighted, it's the standard 2×2 RD with a binomial Wald CI."""
    keep = [treat_col, spec.event_col] + ([cluster_col] if cluster_col and cluster_col in data.columns else [])
    d = data[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=[treat_col, spec.event_col])
    if d[treat_col].nunique() < 2 or len(d) < 20 or d[spec.event_col].nunique() < 2:
        return None
    if weights is not None:
        w = pd.Series(np.asarray(weights), index=data.index).reindex(d.index).values
        Xd = sm.add_constant(d[[treat_col]], has_constant="add")
        # freq_weights inflate N, so (as in the IPTW logistic path) take a cluster-robust
        # sandwich SE — each independent patient its own cluster — for honest inference.
        grp = d[cluster_col].values if (cluster_col and cluster_col in d.columns) else np.arange(len(d))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sm.GLM(d[spec.event_col], Xd, family=sm.families.Gaussian(),
                             freq_weights=w).fit(cov_type="cluster", cov_kwds={"groups": grp})
        except Exception as ex:
            return {"error": str(ex)}
        rd = float(res.params[treat_col])
        ci = res.conf_int().loc[treat_col]
        lo, hi, p = float(ci[0]), float(ci[1]), float(res.pvalues[treat_col])
        et, ec = _arm_event_rates(d.assign(_w=w), treat_col, spec.event_col, "_w")
        test = "Risk difference (IPTW-weighted, robust SE)"
    else:
        et, ec = _arm_event_rates(d, treat_col, spec.event_col, None)
        n1 = int((d[treat_col] == 1).sum()); n0 = int((d[treat_col] == 0).sum())
        rd = et - ec
        se = np.sqrt(et * (1 - et) / n1 + ec * (1 - ec) / n0) if n1 and n0 else np.nan
        lo, hi = rd - 1.96 * se, rd + 1.96 * se
        # two-proportion z-test p-value
        pbar = (et * n1 + ec * n0) / (n1 + n0)
        se0 = np.sqrt(pbar * (1 - pbar) * (1 / n1 + 1 / n0)) if n1 and n0 else np.nan
        p = float(2 * scipy_stats.norm.sf(abs(rd / se0))) if se0 and se0 > 0 else np.nan
        test = "Risk difference (2×2)"
    return {
        "estimate_name": "Risk Difference", "estimate": float(rd),
        "ci_low": float(lo), "ci_high": float(hi), "p_value": float(p), "n": int(len(d)),
        "n_treated": int((d[treat_col] == 1).sum()), "n_control": int((d[treat_col] == 0).sum()),
        "event_treated": et, "event_control": ec,
        "risk_difference": float(rd), "rd_ci_low": float(lo), "rd_ci_high": float(hi),
        "test": test, "e_value": None,
    }


def run_twobytwo(data, spec: OutcomeSpec, treat_col="T", method="chi2"):
    d = data[[treat_col, spec.event_col]].dropna()
    if d[treat_col].nunique() < 2 or d[spec.event_col].nunique() < 2:
        return None
    a = int(((d[treat_col] == 1) & (d[spec.event_col] == 1)).sum())
    b = int(((d[treat_col] == 1) & (d[spec.event_col] == 0)).sum())
    c = int(((d[treat_col] == 0) & (d[spec.event_col] == 1)).sum())
    e = int(((d[treat_col] == 0) & (d[spec.event_col] == 0)).sum())
    table = np.array([[a, b], [c, e]])
    p_treat = a / (a + b) if (a + b) else np.nan
    p_ctrl = c / (c + e) if (c + e) else np.nan
    if method == "fisher":
        or_, p = scipy_stats.fisher_exact(table)
        test_name = "Fisher's exact test"
    else:
        chi2, p, _, _ = scipy_stats.chi2_contingency(table, correction=True)
        or_ = (a * e) / (b * c) if (b * c) else np.nan
        test_name = "Pearson χ² test"
    # OR 95% CI via Woolf (log) approximation
    if all(v > 0 for v in (a, b, c, e)):
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / e)
        lo, hi = np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se)
    else:
        lo = hi = np.nan
    risk_diff = p_treat - p_ctrl
    rd_se = np.sqrt(p_treat * (1 - p_treat) / (a + b) + p_ctrl * (1 - p_ctrl) / (c + e))
    rd_lo, rd_hi = risk_diff - 1.96 * rd_se, risk_diff + 1.96 * rd_se
    return {
        "estimate_name": "Odds Ratio", "estimate": float(or_), "ci_low": float(lo), "ci_high": float(hi),
        "p_value": float(p), "n": int(len(d)),
        "n_treated": int(a + b), "n_control": int(c + e),
        "event_treated": float(p_treat), "event_control": float(p_ctrl),
        "risk_difference": float(risk_diff), "rd_ci_low": float(rd_lo), "rd_ci_high": float(rd_hi),
        "test": test_name, "e_value": e_value(float(or_), float(lo), float(hi)),
    }


# ──────────────────────────────────────────────────────────────────────────
# Kaplan–Meier curves (data for plotting)
# ──────────────────────────────────────────────────────────────────────────
def km_curves(data, spec: OutcomeSpec, treat_col="T", weights=None):
    """Per-arm Kaplan–Meier survival curves. Under IPTW, pass the weights so the curves
    are the weighted (pseudo-population) survival — consistent with the weighted HR —
    rather than crude. PSM is already adjusted via the matched sample passed in."""
    if weights is not None:
        data = data.copy()
        w_arr = np.asarray(weights)
        data["_w"] = w_arr if len(w_arr) == len(data) else np.resize(w_arr, len(data))
    sub = _prep_survival(data, spec, treat_col, extra_cols=["_w"] if weights is not None else None)
    curves = {}
    for tval, name in [(1, "Treated"), (0, "Untreated")]:
        s = sub[sub[treat_col] == tval]
        if len(s) < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(s[spec.time_col], s[spec.event_col], label=name,
                weights=s["_w"] if weights is not None else None)
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        curves[name] = {
            "t": sf.index.values.tolist(),
            "s": sf.iloc[:, 0].values.tolist(),
            "lo": ci.iloc[:, 0].values.tolist(),
            "hi": ci.iloc[:, 1].values.tolist(),
            "n": int(len(s)),
        }
    return curves


# ──────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ──────────────────────────────────────────────────────────────────────────
def prepare_cohort(df, cfg: AnalysisConfig, spec: OutcomeSpec, subgroup_mask=None,
                   compute_overlap=True):
    """Apply subgroup + outcome restriction, then the chosen adjustment.
    Returns (analysis_df, weights_or_None, balance_dict)."""
    work = df.copy()
    if subgroup_mask is not None:
        work = work[subgroup_mask].copy()
    if spec.restrict_col and spec.restrict_col in work.columns:
        work = work[work[spec.restrict_col] == 1].copy()

    covars = [c for c in cfg.covariates if c in work.columns]
    for c in covars:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    # drop covariates entirely missing within this subgroup (median would be NaN)
    covars = [c for c in covars if work[c].notna().any()]
    if covars:
        work[covars] = work[covars].fillna(work[covars].median())
    # drop covariates with no variance (constant) — useless and break the PS fit
    covars = [c for c in covars if work[c].nunique() > 1]

    smd_before = compute_smd(work, covars, cfg.treatment_col) if covars else pd.Series(dtype=float)
    weights = None

    # Single treatment class → nothing to compare; signal cleanly instead of crashing.
    if work[cfg.treatment_col].nunique() < 2:
        balance = {"before": smd_before, "after": smd_before, "method": cfg.adjustment,
                   "overlap": None, "covars": covars,
                   "note": "only one treatment group present — no contrast to estimate"}
        return work, None, balance

    # No usable covariates (all missing/constant in this subgroup): fall back to
    # an unadjusted estimate rather than crashing the propensity model.
    if not covars or cfg.adjustment == "unadjusted":
        balance = {"before": smd_before, "after": smd_before,
                   "method": "unadjusted" if not covars and cfg.adjustment != "unadjusted"
                   else cfg.adjustment, "overlap": None, "covars": covars,
                   "note": ("no usable covariates in this subgroup — unadjusted"
                            if not covars and cfg.adjustment != "unadjusted" else None)}
        return work, None, balance

    # positivity/overlap is only meaningful when a propensity model is used
    overlap = None
    if compute_overlap and cfg.adjustment in ("psm", "iptw") and work[cfg.treatment_col].nunique() == 2:
        try:
            overlap = ps_overlap(work, covars, cfg.treatment_col)
        except Exception:
            overlap = None

    if cfg.adjustment == "psm":
        matched, _ = propensity_match_1toN(work, covars, cfg.treatment_col,
                                           N=cfg.match_ratio, caliper_mad=cfg.caliper_mad)
        smd_after = compute_smd(matched, covars, cfg.treatment_col) if len(matched) else smd_before
        balance = {"before": smd_before, "after": smd_after, "method": "Propensity matching",
                   "overlap": overlap, "covars": covars}
        return matched, None, balance

    if cfg.adjustment == "iptw":
        weights, _ = iptw_weights(work, covars, cfg.treatment_col,
                                  stabilized=cfg.iptw_stabilized, trim=cfg.iptw_trim)
        smd_after = compute_smd(work, covars, cfg.treatment_col, weights=weights)
        balance = {"before": smd_before, "after": smd_after, "method": "IPTW",
                   "overlap": overlap, "covars": covars}
        return work, weights, balance

    # covariate-adjusted: no reweighting/matching
    balance = {"before": smd_before, "after": smd_before, "method": cfg.adjustment,
               "overlap": None, "covars": covars}
    return work, None, balance


def run_analysis(df, cfg: AnalysisConfig, spec: OutcomeSpec, subgroup_mask=None, light=False):
    """Full pipeline for one outcome under one configuration. Returns a result dict.
    light=True skips KM curves, the PH test, and overlap (used by the permutation
    falsification test where only the point estimate is needed)."""
    work, weights, balance = prepare_cohort(df, cfg, spec, subgroup_mask,
                                            compute_overlap=not light)
    # use the covariate set actually retained by prepare_cohort (after dropping
    # missing/constant ones in this subgroup)
    covars = balance.get("covars", [c for c in cfg.covariates if c in work.columns])
    adjusted = weights is not None or balance.get("method") == "Propensity matching"

    cluster = cfg.id_col if (adjusted and balance.get("method") == "Propensity matching") else None
    if spec.kind == "survival":
        adj_cov = covars if cfg.adjustment == "covariate" else None
        if cfg.survival_test == "logrank":
            res = run_logrank(work, spec, cfg.treatment_col)
        else:
            res = run_cox(work, spec, cfg.treatment_col, weights=weights,
                          adjust_covariates=adj_cov, compute_ph=not light,
                          cluster_col=cluster)
        # weight the curves under IPTW so the displayed survival matches the weighted HR
        km = None if light else km_curves(work, spec, cfg.treatment_col, weights=weights)
    else:  # binary
        if cfg.binary_test == "aipw":
            # Doubly-robust ATE: self-contained (its own PS + outcome models), so it runs
            # on the full subgroup cohort and ignores the adjustment dropdown/weights.
            sg = df if subgroup_mask is None else df[subgroup_mask]
            if spec.restrict_col and spec.restrict_col in sg.columns:
                sg = sg[sg[spec.restrict_col] == 1]
            res = aipw_riskdiff(sg, spec, covars, cfg.treatment_col)
        elif cfg.binary_test in ("chi2", "fisher"):
            res = run_twobytwo(work, spec, cfg.treatment_col, method=cfg.binary_test)
        elif cfg.binary_test == "riskdiff":
            res = run_riskdiff(work, spec, cfg.treatment_col, weights=weights,
                               cluster_col=cluster)
        else:  # logistic
            adj_cov = covars if cfg.adjustment == "covariate" else None
            res = run_logistic(work, spec, cfg.treatment_col, covariates=adj_cov,
                               weights=weights, cluster_col=cluster)
        km = None

    if not res or "estimate" not in res:
        return {"ok": False, "outcome": spec.label, "balance": balance,
                "error": (res or {}).get("error", "insufficient data"),
                "n_analyzed": int(len(work))}

    # Absolute risk difference + number-needed-to-treat/harm (event-rate based)
    art, arc = res.get("event_treated"), res.get("event_control")
    abs_rd = nnt = None
    nnt_kind = None
    if art is not None and arc is not None and np.isfinite(art) and np.isfinite(arc):
        abs_rd = float(art - arc)
        if abs(abs_rd) > 1e-9:
            nnt = float(1.0 / abs(abs_rd))
            # Is MORE of this event bad? Death/mortality (benefit_direction "lower",
            # not reversed) → yes. But for REVERSED time-to-event outcomes the event is
            # the *good* thing (ICU discharge, ventilator liberation), so more of it is
            # a benefit → NNT, not NNH.
            harmful_event = (spec.benefit_direction == "lower") and not spec.reverse
            worse = abs_rd > 0 if harmful_event else abs_rd < 0
            nnt_kind = "NNH" if worse else "NNT"

    # Describe what the displayed KM curves represent: under PSM they are computed on
    # the matched sample and under IPTW they are IPTW-weighted (both adjusted), but under
    # covariate adjustment / unadjusted the curves are CRUDE — the covariate adjustment
    # applies to the hazard ratio, not the curves. Surface this so the plot isn't
    # mistaken for an adjusted survival curve when it is not.
    if km:
        _m = balance.get("method")
        if _m == "Propensity matching":
            km_basis = "Curves computed on the propensity-matched sample (adjusted)."
        elif _m == "IPTW":
            km_basis = "Curves are IPTW-weighted (adjusted)."
        else:
            km_basis = ("Curves are unadjusted (crude); the covariate adjustment applies to "
                        "the hazard ratio, not the curves."
                        if cfg.adjustment == "covariate"
                        else "Curves are unadjusted (crude).")
    else:
        km_basis = None

    res.update({
        "ok": True, "outcome": spec.label, "outcome_key": spec.key,
        "reverse": spec.reverse, "balance": balance, "km_basis": km_basis,
        "km": km, "adjustment": cfg.adjustment, "n_analyzed": int(len(work)),
        "n_unbalanced_before": int((balance["before"] > SMD_THRESHOLD).sum()),
        "n_unbalanced_after": int((balance["after"] > SMD_THRESHOLD).sum()),
        "n_covariates": len(covars),
        "abs_risk_diff": abs_rd, "nnt": nnt, "nnt_kind": nnt_kind,
        "overlap": balance.get("overlap"),
    })
    return res


def interaction_test(df, cfg: AnalysisConfig, spec: OutcomeSpec,
                     mask_a, mask_b, name_a="A", name_b="B"):
    """Formal effect-modification test between two subgroups: fit one outcome model
    on both subgroups with treatment, a subgroup indicator G, their interaction
    T×G, and (penalized) covariates. The T×G term's p-value tests whether the
    treatment effect differs across subgroups (a stronger claim than comparing
    separate subgroup estimates)."""
    a = df[mask_a].copy(); a["_G"] = 0
    b = df[mask_b].copy(); b["_G"] = 1
    work = pd.concat([a, b], ignore_index=True)
    if spec.restrict_col and spec.restrict_col in work.columns:
        work = work[work[spec.restrict_col] == 1].copy()
    tcol = cfg.treatment_col
    covars = [c for c in cfg.covariates if c in work.columns]
    for c in covars:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work[covars] = work[covars].fillna(work[covars].median())
    if covars:
        covars = [c for c in covars if work[c].nunique() > 1]
        work[covars] = (work[covars] - work[covars].mean()) / work[covars].std(ddof=0).replace(0, np.nan)
        work[covars] = work[covars].fillna(0.0)
        # drop near-collinear covariates (e.g. SOFA total vs its components)
        corr = work[covars].corr().abs()
        drop = set()
        for i, ci in enumerate(covars):
            for cj in covars[i + 1:]:
                if cj not in drop and corr.loc[ci, cj] > 0.9:
                    drop.add(cj)
        covars = [c for c in covars if c not in drop]
    work["_TxG"] = work[tcol] * work["_G"]
    if work["_G"].nunique() < 2 or work[tcol].nunique() < 2:
        return {"ok": False, "error": "need two subgroups with both treatment arms"}

    try:
        if spec.kind == "survival":
            cols = [spec.time_col, spec.event_col, tcol, "_G", "_TxG"] + covars
            sub = work[cols].dropna()
            sub = sub[sub[spec.time_col] > 0]
            ev = sub.groupby("_G")[spec.event_col].sum()
            if len(sub) < 40 or ev.min() < 3:
                return {"ok": False, "error": "too few events for an interaction test"}
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(sub, duration_col=spec.time_col, event_col=spec.event_col)
            coef = float(cph.summary.loc["_TxG", "coef"])
            p = float(cph.summary.loc["_TxG", "p"])
            ratio = float(np.exp(coef))
            name = "HR ratio (T×G)"
        else:
            y = pd.to_numeric(work[spec.event_col], errors="coerce")
            Xc = [tcol, "_G", "_TxG"] + covars
            d = pd.concat([y.rename("_y"), work[Xc]], axis=1).dropna()
            if len(d) < 40 or d["_y"].nunique() < 2:
                return {"ok": False, "error": "insufficient data for interaction test"}
            Xd = sm.add_constant(d[Xc], has_constant="add")
            res = sm.Logit(d["_y"], Xd).fit(disp=0)
            coef = float(res.params["_TxG"]); p = float(res.pvalues["_TxG"])
            ratio = float(np.exp(coef)); name = "OR ratio (T×G)"
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "p_value": p, "ratio": ratio, "ratio_name": name,
            "name_a": name_a, "name_b": name_b,
            "significant": p < 0.05, "n": int(len(work))}


def falsification_test(df, cfg: AnalysisConfig, spec: OutcomeSpec, subgroup_mask=None,
                       n_rounds=10, seed=42):
    """Placebo/negative-control test: randomly re-assign treatment (same prevalence)
    and re-run the full adjustment + estimation. A valid pipeline should produce a
    null distribution (estimates near the no-effect value, ~5% significant at 0.05).
    Systematic departure signals the machinery is manufacturing effects."""
    work = df.copy()
    if subgroup_mask is not None:
        work = work[subgroup_mask].copy()
    if spec.restrict_col and spec.restrict_col in work.columns:
        work = work[work[spec.restrict_col] == 1].copy()
    work = work.reset_index(drop=True)
    tcol = cfg.treatment_col
    n = len(work)
    n_treat = int((work[tcol] == 1).sum())
    if n < 40 or n_treat < 5 or n_treat >= n:
        return {"ok": False, "error": "cohort too small for a stable placebo test"}

    rng = np.random.default_rng(seed)
    null_val = 0.0 if spec.kind == "binary" and cfg.binary_test in ("riskdiff", "aipw") else 1.0
    ests, pvals = [], []
    ename = None
    for _ in range(n_rounds):
        perm = np.zeros(n, dtype=int)
        perm[rng.choice(n, size=n_treat, replace=False)] = 1
        w2 = work.copy()
        w2[tcol] = perm
        r = run_analysis(w2, cfg, spec, subgroup_mask=None, light=True)
        if r.get("ok") and np.isfinite(r.get("estimate", np.nan)):
            ests.append(float(r["estimate"]))
            pvals.append(float(r["p_value"]))
            if ename is None:
                ename = r.get("estimate_name")   # the engine's own estimand label
    if not ests:
        return {"ok": False, "error": "placebo runs did not converge"}

    ests = np.array(ests)
    frac_sig = float(np.mean(np.array(pvals) < 0.05))
    # log-scale centering for ratio estimands
    if null_val == 1.0:
        center = float(np.exp(np.median(np.log(ests))))
        spread = float(np.exp(np.std(np.log(ests))))
    else:
        center = float(np.median(ests))
        spread = float(np.std(ests))
    well_calibrated = (frac_sig <= 0.20) and (abs(np.log(center) if null_val == 1 else center) < 0.15)
    return {
        "ok": True, "null_value": null_val, "n_rounds": len(ests),
        "median_estimate": center, "spread": spread,
        "min": float(ests.min()), "max": float(ests.max()),
        "frac_significant": frac_sig, "estimates": ests.tolist(),
        "well_calibrated": bool(well_calibrated),
        # Use the engine's actual estimand label (Risk Difference under riskdiff/aipw,
        # Odds Ratio under logistic/χ²/Fisher, Hazard Ratio for survival) so the placebo
        # plot's axis and null line are consistent with what was permuted.
        "estimate_name": ename or ("Hazard Ratio" if spec.kind == "survival" else "Odds Ratio"),
    }
