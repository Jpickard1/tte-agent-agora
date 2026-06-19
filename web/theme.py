"""Shared visual theme for the tteEngine results app (#104).

Self-contained adaptation of emulaTTE's web/app.py look (NOT imported from it):
IBM Plex Sans+Mono, plum/teal/amber palette, wide layout, rounded card containers
with soft shadows, styled metric cards + plum tabs, eyebrow micro-labels, and the
live pulse dot. Plus a few tteEngine-specific marks (pipeline step cards, agreement
badges). HTML/CSS helpers only — no Streamlit import here so it stays light.
"""

from __future__ import annotations

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
:root{
  --canvas:#e7e2d8; --main-bg:#f7f4ef; --sidebar-bg:#ece7df; --surface:#faf8f4;
  --surface-2:#f3efe8; --ink:#2c2935; --ink-2:#4a4656; --muted:#8b8598;
  --faint:#aaa3b2; --hairline:#e6dfd3; --border:#e0d9cc;
  --plum:#6b5ea6; --plum-deep:#50457f; --plum-soft:#efeaf7; --plum-line:#ddd3ee;
  --teal:#3f8f86; --teal-soft:#e7f1ee; --teal-line:#cfe5df; --amber:#b8823c;
}
html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family:'IBM Plex Sans', sans-serif !important; color:var(--ink); }
.stApp { background:var(--canvas); }
.main .block-container { background:var(--main-bg); border:1px solid var(--hairline);
  border-radius:8px; padding:26px 34px 40px; margin-top:14px;
  box-shadow:0 1px 3px rgba(50,38,70,.07), 0 22px 50px -28px rgba(50,38,70,.28); }
[data-testid="stSidebar"] { background:var(--sidebar-bg); border-right:1px solid var(--hairline); }
h1,h2,h3,h4 { font-weight:600 !important; letter-spacing:-0.01em; color:var(--ink); }
code, kbd, pre, [data-testid="stMetricValue"], .stCode {
  font-family:'IBM Plex Mono', monospace !important; font-variant-numeric:tabular-nums; }
.stButton>button { font-family:'IBM Plex Sans',sans-serif; font-weight:600; border-radius:8px;
  border:1px solid var(--border); background:var(--surface); color:var(--ink-2); transition:all .14s ease; }
.stButton>button:hover { border-color:var(--plum); color:var(--plum-deep); }
.stButton>button[kind="primary"]{ background:var(--plum); color:#faf8f4; border-color:var(--plum);
  box-shadow:0 2px 8px -1px rgba(107,94,166,.5); }
.stButton>button[kind="primary"]:hover{ background:var(--plum-deep); color:#fff; }
[data-baseweb="input"], [data-baseweb="select"]>div, .stTextInput input,
[data-testid="stTextInputRootElement"] { background:var(--surface-2)!important; border-radius:8px!important; }
[data-testid="stMetric"]{ background:var(--surface); border:1px solid var(--hairline);
  border-radius:12px; padding:14px 16px; }
[data-testid="stMetricValue"]{ font-size:30px; font-weight:600; color:var(--ink); }
[data-testid="stMetricLabel"]{ font-family:'IBM Plex Mono',monospace; text-transform:uppercase;
  letter-spacing:.12em; font-size:10px; color:var(--muted); }
hr, [data-testid="stDivider"]{ border-color:var(--hairline)!important; }
.stTabs [data-baseweb="tab-list"]{ gap:34px; border-bottom:1px solid var(--hairline); margin-bottom:14px; }
.stTabs [data-baseweb="tab"]{ font-family:'IBM Plex Mono',monospace; font-size:14px;
  letter-spacing:.06em; color:var(--muted); padding:8px 4px; }
.stTabs [aria-selected="true"]{ color:var(--plum-deep)!important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--plum)!important; }
[data-testid="stDataFrame"], [data-testid="stTable"]{ border-radius:10px; }
.eyebrow{ font-family:'IBM Plex Mono',monospace; text-transform:uppercase;
  letter-spacing:.18em; font-size:10px; color:var(--muted); }
.eb-header{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; }
.eb-tag{ font-family:'IBM Plex Mono',monospace; font-size:11px; color:var(--muted);
  border:1px solid var(--border); border-radius:6px; padding:2px 8px; }
.eb-crumb{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--ink-2); }
.eb-core{ margin-left:auto; font-family:'IBM Plex Mono',monospace; font-size:11px;
  color:var(--ink-2); background:var(--surface); border:1px solid var(--hairline);
  border-radius:7px; padding:4px 10px; display:flex; align-items:center; gap:7px; }
.dot{ width:8px;height:8px;border-radius:50%;display:inline-block; }
.dot-live{ background:var(--teal); animation:pulse 1.6s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.8)} }
/* tteEngine: pipeline step cards (overview guided path) */
.pipe{ display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; margin:6px 0 4px; }
.pcard{ flex:1; min-width:150px; background:var(--surface); border:1px solid var(--hairline);
  border-radius:12px; padding:13px 15px; box-shadow:0 1px 3px rgba(50,38,70,.06); }
.pcard .pn{ font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--plum); }
.pcard .pt{ font-weight:600; font-size:15px; margin:3px 0 2px; }
.pcard .pd{ font-size:12.5px; color:var(--ink-2); line-height:1.35; }
.parrow{ display:flex; align-items:center; color:var(--faint); font-size:18px; }
/* agreement badge */
.badge{ font-family:'IBM Plex Mono',monospace; font-size:11px; border-radius:11px;
  padding:2px 10px; border:1px solid; }
.b-conc{ color:#1a7a4d; background:#1a7a4d14; border-color:#1a7a4d55; }
.b-disc{ color:#c0392b; background:#c0392b14; border-color:#c0392b55; }
.b-inc { color:#4a4656; background:#4a465614; border-color:#4a465655; }
.card{ background:var(--surface); border:1px solid var(--hairline); border-radius:12px;
  padding:14px 16px; box-shadow:0 1px 3px rgba(50,38,70,.06); margin-bottom:10px; }
/* guided walkthrough stepper (sidebar) */
.step{ display:flex; align-items:center; gap:10px; padding:5px 0; font-size:14px; }
.step .node{ width:22px;height:22px;border-radius:7px;display:flex;align-items:center;
  justify-content:center;font-family:'IBM Plex Mono',monospace;font-size:11px;font-weight:600;flex:none; }
.step.done .node{ background:var(--teal); color:#faf8f4; }
.step.active .node{ background:var(--surface); border:2px solid var(--plum); color:var(--plum); }
.step.locked .node{ background:var(--surface-2); border:1px solid var(--border); color:var(--faint); }
.step.done .lbl{ color:var(--ink-2); } .step.active .lbl{ color:var(--ink); font-weight:600; }
.step.locked .lbl{ color:var(--faint); }
</style>
"""

_LOGO = ("<svg width='28' height='28' viewBox='0 0 30 30'>"
         "<rect width='30' height='30' rx='9' fill='#6b5ea6'/>"
         "<circle cx='15' cy='15' r='8.5' fill='none' stroke='#faf8f4' stroke-width='1.6'/>"
         "<circle cx='15' cy='15' r='4.5' fill='none' stroke='#faf8f4' stroke-width='1.6'/>"
         "<circle cx='15' cy='15' r='1.6' fill='#faf8f4'/></svg>")


def header_html(crumb: str, *, source: str = "persisted corpus") -> str:
    return (
        "<div class='eb-header'>" + _LOGO +
        "<span style='font-size:22px;font-weight:600;letter-spacing:-.01em'>"
        "<span style='color:var(--ink)'>tte</span><span style='color:var(--plum)'>Engine</span></span>"
        "<span class='eb-tag'>Target Trial Emulation</span>"
        f"<span class='eb-crumb'>{crumb}</span>"
        f"<span class='eb-core'><span class='dot dot-live'></span>{source}</span>"
        "</div>"
    )


def badge_html(agreement: str | None) -> str:
    cls = {"concordant": "b-conc", "discordant": "b-disc"}.get(agreement or "", "b-inc")
    return f"<span class='badge {cls}'>{agreement or 'inconclusive'}</span>"


def stepper_html(steps: list[str], active: int = 0) -> str:
    """emulaTTE-style guided stepper with progress: steps before `active` are done
    (teal check), `active` is highlighted (plum), later steps are locked."""
    rows = []
    for i, label in enumerate(steps):
        cls = "done" if i < active else ("active" if i == active else "locked")
        node = "✓" if i < active else str(i + 1)
        rows.append(f"<div class='step {cls}'><div class='node'>{node}</div>"
                    f"<div class='lbl'>{label}</div></div>")
    return "".join(rows)


def pipeline_html() -> str:
    steps = [
        ("01 · Read", "ClinicalTrials.gov trial", "Parse a real RCT's protocol — eligibility, arms, outcome, time-zero."),
        ("02 · Extract", "MIMIC-IV · eICU · MGB", "Pull the matching cohort from each ICU database into one common format."),
        ("03 · Emulate", "Target trial emulation", "Build arms with a landmark time-zero, adjust confounding (IPTW/PSM), estimate the effect."),
        ("04 · Compare", "vs the real RCT", "Concordant or discordant with the trial's reported result — with the WHY."),
    ]
    cards = []
    for i, (pn, pt, pd) in enumerate(steps):
        cards.append(f"<div class='pcard'><div class='pn'>{pn}</div><div class='pt'>{pt}</div>"
                     f"<div class='pd'>{pd}</div></div>")
        if i < len(steps) - 1:
            cards.append("<div class='parrow'>→</div>")
    return "<div class='pipe'>" + "".join(cards) + "</div>"
