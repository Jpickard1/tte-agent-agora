"""analysis: estimand-aware TTE engine (#10) + emulated-vs-observed benchmark (#11).

#10 wraps trialsim app/trialsim/tte_engine.py (PSM/IPTW/Cox/AIPW/KM/log-rank/
E-values/robustness/replication); consumes the #9 analysis frame, returns a
TTEResult. #11 ports trialsim compare.py (ctgov resultsSection -> RR/RD +
concordance) and adds the batch harness over trials x DBs. Owner: probe.

The engine (engine.py) is a self-contained PORT of trialsim's tte_engine.py and
needs the `analysis` extra (lifelines/statsmodels/scikit-learn); the typed
entrypoint run_tte + TTEResult are import-light wrappers.
"""

from .runner import BalanceRow, TTEResult, add_treatment_indicator, run_tte
from .compare import compare_trial, concordance, parse_reported_effect
from .benchmark import benchmark_trials, run_benchmark

__all__ = [
    "run_tte",
    "TTEResult",
    "BalanceRow",
    "add_treatment_indicator",
    "parse_reported_effect",
    "concordance",
    "compare_trial",
    "run_benchmark",
    "benchmark_trials",
]
