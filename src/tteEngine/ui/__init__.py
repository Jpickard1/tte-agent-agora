"""ui: the results dashboard (#49) + clinician Trial Emulation Cards (#40).

Pure, import-light data layer (build_cards / build_dashboard) that any front-end
renders — the Streamlit app (web/results_app.py) is just a thin renderer over it,
needing no matplotlib. Reads the persisted corpus via
contracts.io.load_comparisons_jsonl, so it works on the live >1k/>10k run.
"""

from .cards import TrialEmulationCard, build_cards
from .context_panel import corpus_context_summary, index_context, why_for
from .dashboard import (
    DashboardModel,
    build_dashboard,
    ctgov_url,
    group_by_trial,
    trial_table,
)

__all__ = [
    "TrialEmulationCard",
    "build_cards",
    "DashboardModel",
    "build_dashboard",
    "index_context",
    "why_for",
    "corpus_context_summary",
    "ctgov_url",
    "trial_table",
    "group_by_trial",
]
