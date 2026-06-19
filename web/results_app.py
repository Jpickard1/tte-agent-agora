"""Results gallery (#104) — the clinician-facing capstone, modeled on emulaTTE.

Three clear views via top tabs (not a function list):
  1. OVERVIEW   — what the system does, plain-language + a guided pipeline.
  2. ALL TRIALS — corpus headline (concordance + CI, I²/τ², calibration) + a
                  sortable/filterable table of every (trial, dataset), each row
                  linking to its real ClinicalTrials.gov study.
  3. PER-TRIAL  — pick a trial: emulated-vs-RCT per dataset, forest, the WHY
                  context, and its ctgov link.

Polished theme copied in from emulaTTE (web/theme.py — self-contained, NOT
imported). Reads the persisted corpus via contracts.io, so it shows real data
automatically once the live MIMIC/eICU run lands. Import-light: altair, no matplotlib.

Run (needs the `web` extra):
    TTE_CORPUS_JSONL=corpus.jsonl streamlit run web/results_app.py
    # optional: TTE_CONTEXT_JSONL=context.jsonl  TTE_CATALOG_CSV=catalog.csv
"""

from __future__ import annotations

import csv
import os

import altair as alt
import pandas as pd
import streamlit as st

import theme
from tteEngine.contracts.io import load_comparisons_jsonl
from tteEngine.ui import build_dashboard, ctgov_url, group_by_trial, trial_table

_RATIO = {"RR", "OR", "HR"}
CTGOV = "https://clinicaltrials.gov/study/"


def _sepsis_ncts(catalog_csv: str) -> set[str]:
    out: set[str] = set()
    if catalog_csv and os.path.exists(catalog_csv):
        with open(catalog_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("is_sepsis", "")).strip().lower() in ("true", "1", "yes"):
                    out.add(row["nct_id"])
    return out


def _forest_chart(rows):
    df = pd.DataFrame([r.model_dump() if hasattr(r, "model_dump") else r for r in rows])
    if df.empty:
        return None
    ratio = (df["measure"].isin(_RATIO)).mean() >= 0.5
    scale = alt.Scale(type="log") if ratio else alt.Scale(type="linear")
    base = alt.Chart(df)
    color = alt.Color("agreement:N", scale=alt.Scale(
        domain=["concordant", "discordant", "inconclusive"],
        range=["#3f8f86", "#c0392b", "#8b8598"]), legend=alt.Legend(title="agreement"))
    rule = base.mark_rule().encode(
        x=alt.X("ci_low:Q", scale=scale, title=f"{df['measure'].iloc[0]} (emulated · log)"),
        x2="ci_high:Q", y=alt.Y("label:N", sort=None, title=None), color=color)
    pt = base.mark_point(filled=True, size=60).encode(
        x="estimate:Q", y=alt.Y("label:N", sort=None), color=color,
        tooltip=["label", "measure", "estimate", "ci_low", "ci_high", "observed_estimate", "agreement"])
    return (rule + pt).properties(height=min(30 * len(df) + 40, 820))


def _calibration_chart(points):
    if not points:
        return None
    df = pd.DataFrame(points)
    sc = alt.Scale(type="log")
    lo = min(df.emulated.min(), df.observed.min()) * 0.8
    hi = max(df.emulated.max(), df.observed.max()) * 1.25
    ident = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(
        color="#8b8598", strokeDash=[4, 4]).encode(x=alt.X("x:Q", scale=sc), y=alt.Y("x:Q", scale=sc))
    pts = alt.Chart(df).mark_circle(size=70).encode(
        x=alt.X("emulated:Q", scale=sc, title="emulated effect"),
        y=alt.Y("observed:Q", scale=sc, title="observed (RCT) effect"),
        color=alt.Color("in_ci:N", title="CI covers observed",
                        scale=alt.Scale(domain=[True, False], range=["#3f8f86", "#c0392b"])),
        tooltip=["emulated", "observed", "in_ci"])
    return (ident + pts).properties(height=360, width=360)


def _eyebrow(text):
    st.markdown(f"<div class='eyebrow'>{text}</div>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- views
def view_overview(m):
    _eyebrow("OVERVIEW")
    st.markdown("### Can we recover real clinical-trial results from ICU records?")
    st.markdown(
        "**tteEngine** reads a randomized trial from ClinicalTrials.gov, rebuilds an "
        "equivalent *target trial* inside real ICU databases (MIMIC-IV, eICU, MGB), emulates "
        "it with causal methods, and checks whether the emulated effect **agrees with the "
        "real trial** — and explains *why* when it doesn't.")
    st.markdown(theme.pipeline_html(), unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.markdown("<div class='card'><b>How to read it</b><br>"
                    "<span style='font-size:13px;color:var(--ink-2)'>"
                    "<b>Concordant</b> (teal) = emulation matches the trial's direction. "
                    "<b>Calibration slope→1.0</b> and <b>coverage→95%</b> mean the emulated "
                    "effects line up with the real ones. The classic check: confounded crude "
                    "estimates can show <i>harm</i>, while the adjusted emulation recovers the "
                    "trial's <i>benefit</i>.</span></div>", unsafe_allow_html=True)
    with b:
        st.markdown("<div class='card'><b>What's in the corpus</b><br>"
                    "<span style='font-size:13px;color:var(--ink-2)'>"
                    f"<b>{m.n_total}</b> emulations · <b>{m.n_sepsis}</b> sepsis · across ICU "
                    "datasets. Open <b>All trials</b> for the headline numbers + the full table, "
                    "or <b>Per-trial</b> to inspect any single trial vs its real RCT.</span></div>",
                    unsafe_allow_html=True)
    st.caption("Showing the persisted corpus — real MIMIC/eICU numbers appear automatically once the live run lands.")


def view_all_trials(m):
    _eyebrow("CORPUS SUMMARY")
    rate, pe, cal = m.concordance.get("rate"), m.pooled, m.calibration
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Concordance", f"{rate:.0%}" if rate is not None else "—",
              help=f"{m.concordance.get('n_concordant')}/{m.concordance.get('n_comparable')} comparable"
                   + (f" · 95% CI {m.concordance['ci_low']:.0%}–{m.concordance['ci_high']:.0%}"
                      if m.concordance.get("ci_low") is not None else ""))
    c2.metric("Pooled effect", f"{pe.get('estimate'):.2f}" if pe.get("estimate") else "—",
              help=f"95% CI {pe.get('ci_low'):.2f}–{pe.get('ci_high'):.2f}" if pe.get("ci_low") else "")
    c3.metric("Heterogeneity I²", f"{pe.get('i2'):.0%}" if pe.get("i2") is not None else "—")
    c4.metric("Calibration slope", f"{cal.get('slope'):.2f}" if cal.get("slope") is not None else "—",
              help=f"coverage {cal.get('coverage'):.0%}" if cal.get("coverage") is not None else "")
    if m.sepsis_pooled:
        sp = m.sepsis_pooled
        st.caption(f"🩸 Sepsis subgroup (k={sp.get('k')}): pooled {sp.get('estimate'):.2f} "
                   f"(95% CI {sp.get('ci_low'):.2f}–{sp.get('ci_high'):.2f}), I²={sp.get('i2'):.0%}"
                   if sp.get("estimate") is not None else "")
    cs = m.context_summary
    if cs.get("n"):
        st.caption(f"Why-emulable: {cs.get('pct_emulable', 0):.0%} emulable · "
                   f"{cs.get('pct_fully_measurable', 0):.0%} fully measurable")

    left, right = st.columns([3, 2])
    with left:
        _eyebrow("FOREST")
        fc = _forest_chart(m.forest_rows)
        st.altair_chart(fc, use_container_width=True) if fc is not None else st.write("—")
    with right:
        _eyebrow("CALIBRATION")
        cc = _calibration_chart(cal.get("points"))
        st.altair_chart(cc, use_container_width=True) if cc is not None else st.write("—")

    _eyebrow("ALL TRIALS · click a ClinicalTrials.gov link, or inspect one in Per-trial")
    df = pd.DataFrame(trial_table(m))
    q = st.text_input("Filter by NCT id", "", placeholder="e.g. NCT01234567")
    if q:
        df = df[df["nct_id"].str.contains(q, case=False, na=False)]
    st.dataframe(df, use_container_width=True, hide_index=True, column_config={
        "ctgov": st.column_config.LinkColumn("ClinicalTrials.gov", display_text="open ↗"),
        "emulated": st.column_config.NumberColumn(format="%.2f"),
        "observed": st.column_config.NumberColumn(format="%.2f"),
        "score": st.column_config.NumberColumn("emulability", format="%.2f"),
    })


def view_per_trial(m):
    _eyebrow("PER-TRIAL DETAIL")
    groups = group_by_trial(m)
    if not groups:
        st.info("No trials in the corpus yet.")
        return
    nct = st.selectbox("Trial (NCT id)", sorted(groups))
    cards = groups[nct]
    st.markdown(f"### {nct}")
    st.link_button("View on ClinicalTrials.gov ↗", ctgov_url(nct))
    if any(c.is_sepsis for c in cards):
        st.caption("🩸 sepsis trial")

    for c in cards:
        with st.container():
            st.markdown(
                f"<div class='card'><b>{c.dataset}</b> &nbsp; {theme.badge_html(c.agreement)}<br>"
                f"<span style='font-size:13px;color:var(--ink-2)'>{c.verdict}</span></div>",
                unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns(3)
            ci = f" (95% CI {c.ci_low:.2f}–{c.ci_high:.2f})" if c.ci_low is not None else ""
            cc1.metric(f"Emulated {c.measure}", f"{c.emulated_estimate:.2f}", help=ci.strip())
            cc2.metric("Observed (RCT)", f"{c.observed_estimate:.2f}" if c.observed_estimate is not None else "—")
            cc3.metric("n", f"{c.n_treated}+{c.n_control}", help=f"p={c.p_value} · E-value={c.e_value}")
            if c.why:
                w = c.why
                st.markdown("**Why** — " + w.get("why_emulable", ""))
                meas = w.get("measurability") or {}
                bits = (f"measurable {meas.get('n_measurable')} · proxy {meas.get('n_proxy')} · "
                        f"unmeasurable {meas.get('n_unmeasurable')}")
                if w.get("proxy_elements"):
                    bits += " · proxied: " + ", ".join(w["proxy_elements"][:5])
                st.caption(bits)
                if w.get("why_divergent"):
                    st.caption(f"cross-dataset variability — {w['why_divergent']}")

    _eyebrow("THIS TRIAL · emulated effect across datasets")
    trial_rows = [r for r in m.forest_rows if r.label.startswith(nct)]
    fc = _forest_chart(trial_rows)
    st.altair_chart(fc, use_container_width=True) if fc is not None else st.write("—")
    st.caption("Per-patient survival (KM) + CONSORT attrition appear here when the run exports "
               "trial-level detail alongside the summary corpus.")


def main() -> None:
    st.set_page_config(page_title="tteEngine — Trial Emulation Results", layout="wide")
    st.markdown(theme.CSS, unsafe_allow_html=True)

    path = st.sidebar.text_input("Corpus JSONL", value=os.environ.get("TTE_CORPUS_JSONL", ""))
    context_path = st.sidebar.text_input("WHY context JSONL (optional)", value=os.environ.get("TTE_CONTEXT_JSONL", ""))
    catalog = st.sidebar.text_input("Catalog CSV (optional)", value=os.environ.get("TTE_CATALOG_CSV", ""))

    crumb = "ctgov trial → MIMIC/eICU/MGB → emulate → compare to RCT"
    st.markdown(theme.header_html(crumb), unsafe_allow_html=True)
    st.markdown("<hr style='margin:10px 0 18px'>", unsafe_allow_html=True)

    if not path or not os.path.exists(path):
        st.info("Set a **Corpus JSONL** path (from `run_corpus_to_jsonl`) in the sidebar to load results.")
        tab, = st.tabs(["Overview"])
        with tab:
            st.markdown(theme.pipeline_html(), unsafe_allow_html=True)
        st.stop()

    sepsis = _sepsis_ncts(catalog)
    rows = list(load_comparisons_jsonl(path))
    if sepsis and st.sidebar.checkbox("Sepsis trials only", value=False):
        rows = [r for r in rows if r.nct_id in sepsis]
    context_records = None
    if context_path and os.path.exists(context_path):
        from tteEngine.contracts.context import load_context_jsonl
        context_records = list(load_context_jsonl(context_path))
    m = build_dashboard(rows, sepsis_ncts=sepsis, context_records=context_records)

    t_over, t_all, t_trial = st.tabs(["Overview", "All trials", "Per-trial"])
    with t_over:
        view_overview(m)
    with t_all:
        view_all_trials(m)
    with t_trial:
        view_per_trial(m)


if __name__ == "__main__":
    main()
