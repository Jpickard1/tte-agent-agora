"""Results gallery dashboard (#49) — the clinician-facing capstone.

A thin Streamlit renderer over the pure ui.build_dashboard model (no matplotlib;
charts via altair). Reads the persisted corpus with contracts.io, so it works on
the live >1k/>10k MIMIC/eICU run. Sepsis-filterable; per-trial Trial Emulation
Cards (#40) in the gallery.

Run (needs the `web` extra: streamlit, altair):
    TTE_CORPUS_JSONL=corpus.jsonl streamlit run web/results_app.py
    # optional sepsis flags: TTE_CATALOG_CSV=catalog.csv  (the #35 triage catalog)
"""

from __future__ import annotations

import csv
import os

import altair as alt
import pandas as pd
import streamlit as st

from tteEngine.contracts.io import load_comparisons_jsonl
from tteEngine.ui import build_dashboard

_RATIO = {"RR", "OR", "HR"}


def _sepsis_ncts(catalog_csv: str) -> set[str]:
    """Sepsis NCT ids from the #35 catalog.csv (columns nct_id, is_sepsis)."""
    out: set[str] = set()
    if catalog_csv and os.path.exists(catalog_csv):
        with open(catalog_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("is_sepsis", "")).strip().lower() in ("true", "1", "yes"):
                    out.add(row["nct_id"])
    return out


def _forest_chart(rows):
    df = pd.DataFrame([r.model_dump() for r in rows])
    if df.empty:
        return None
    ratio = (df["measure"].isin(_RATIO)).mean() >= 0.5
    base = alt.Chart(df)
    scale = alt.Scale(type="log") if ratio else alt.Scale(type="linear")
    rule = base.mark_rule().encode(
        x=alt.X("ci_low:Q", scale=scale, title=f"{df['measure'].iloc[0]} (emulated)"),
        x2="ci_high:Q", y=alt.Y("label:N", sort=None, title=None),
        color=alt.Color("agreement:N",
                        scale=alt.Scale(domain=["concordant", "discordant", "inconclusive"],
                                        range=["#1a9850", "#d73027", "#4575b4"])))
    pt = base.mark_point(filled=True, size=55).encode(
        x="estimate:Q", y=alt.Y("label:N", sort=None), color="agreement:N",
        tooltip=["label", "measure", "estimate", "ci_low", "ci_high", "observed_estimate", "agreement"])
    return (rule + pt).properties(height=min(28 * len(df) + 40, 900))


def _calibration_chart(points):
    if not points:
        return None
    df = pd.DataFrame(points)
    sc = alt.Scale(type="log")
    lo, hi = min(df.emulated.min(), df.observed.min()) * 0.8, max(df.emulated.max(), df.observed.max()) * 1.25
    ident = alt.Chart(pd.DataFrame({"x": [lo, hi]})).mark_line(color="grey", strokeDash=[4, 4]).encode(
        x=alt.X("x:Q", scale=sc), y=alt.Y("x:Q", scale=sc))
    sc_pts = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("emulated:Q", scale=sc, title="emulated effect"),
        y=alt.Y("observed:Q", scale=sc, title="observed effect"),
        color=alt.Color("in_ci:N", title="CI covers observed"),
        tooltip=["emulated", "observed", "in_ci"])
    return (ident + sc_pts).properties(height=380, width=380)


def main() -> None:
    st.set_page_config(page_title="tteEngine — Trial Emulation Results", layout="wide")
    st.title("🧪 Target Trial Emulation — Results Gallery")
    st.caption("Emulated treatment effects vs the real randomized trials, across ICU datasets.")

    path = st.sidebar.text_input("Corpus JSONL", value=os.environ.get("TTE_CORPUS_JSONL", ""))
    catalog = st.sidebar.text_input("Catalog CSV (#35, optional)", value=os.environ.get("TTE_CATALOG_CSV", ""))
    if not path or not os.path.exists(path):
        st.info("Set a corpus JSONL path (produced by `run_corpus_to_jsonl`) in the sidebar.")
        st.stop()

    sepsis = _sepsis_ncts(catalog)
    rows = list(load_comparisons_jsonl(path))
    sepsis_only = st.sidebar.checkbox("Sepsis trials only", value=False, disabled=not sepsis)
    if sepsis_only:
        rows = [r for r in rows if r.nct_id in sepsis]
    m = build_dashboard(rows, sepsis_ncts=sepsis)

    c1, c2, c3, c4 = st.columns(4)
    rate = m.concordance.get("rate")
    c1.metric("Concordance", f"{rate:.0%}" if rate is not None else "—",
              help=f"{m.concordance.get('n_concordant')}/{m.concordance.get('n_comparable')} comparable")
    pe = m.pooled
    c2.metric("Pooled effect", f"{pe.get('estimate'):.2f}" if pe.get("estimate") else "—",
              help=f"95% CI {pe.get('ci_low'):.2f}–{pe.get('ci_high'):.2f}" if pe.get("ci_low") else "")
    c3.metric("Heterogeneity I²", f"{pe.get('i2'):.0%}" if pe.get("i2") is not None else "—")
    cal = m.calibration
    c4.metric("Calibration slope", f"{cal.get('slope'):.2f}" if cal.get("slope") is not None else "—",
              help=f"coverage {cal.get('coverage'):.0%}" if cal.get("coverage") is not None else "")
    st.caption(f"{m.n_total} emulations · {m.n_sepsis} sepsis · ideal: concordance→100%, slope→1.0, coverage→95%")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Forest — emulated effects")
        fc = _forest_chart(m.forest_rows)
        st.altair_chart(fc, use_container_width=True) if fc is not None else st.write("no rows")
    with right:
        st.subheader("Calibration — emulated vs observed")
        cc = _calibration_chart(cal.get("points"))
        st.altair_chart(cc, use_container_width=True) if cc is not None else st.write("no points")

    st.subheader("Trial Emulation Cards")
    for card in m.cards:
        tag = "🟢" if card.agreement == "concordant" else ("🔴" if card.agreement == "discordant" else "⚪")
        sep = " · 🩸 sepsis" if card.is_sepsis else ""
        with st.expander(f"{tag} {card.nct_id} [{card.dataset}]{sep} — {card.verdict}"):
            ci = f" (95% CI {card.ci_low:.2f}–{card.ci_high:.2f})" if card.ci_low is not None else ""
            st.write(f"**Emulated {card.measure}** = {card.emulated_estimate:.2f}{ci}  "
                     f"vs **observed** {card.observed_estimate}")
            st.write(f"n: {card.n_treated} treated / {card.n_control} control · "
                     f"p={card.p_value} · E-value={card.e_value}")


if __name__ == "__main__":
    main()
