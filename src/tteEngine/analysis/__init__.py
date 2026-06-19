"""analysis: estimand-aware TTE engine (#10) + emulated-vs-observed benchmark (#11).

#10 wraps trialsim app/trialsim/tte_engine.py (PSM/IPTW/Cox/AIPW/KM/log-rank/
E-values/robustness/replication); consumes the #9 analysis frame, returns a
TTEResult. #11 ports trialsim compare.py (ctgov resultsSection -> RR/RD +
concordance) and adds the batch harness over trials x DBs. Owner: probe.

The single seam type is contracts.results.TTEResult (run_tte returns it; the
benchmark consumes it and returns ComparisonResult). engine.py is a self-contained
PORT of trialsim's tte_engine.py and needs the `analysis` extra (lifelines/
statsmodels/scikit-learn); run_tte imports it lazily so this package stays light.
"""

from ..contracts.results import ComparisonResult, EffectMeasure, TTEResult
from .runner import add_treatment_indicator, run_tte
from .compare import compare_trial, parse_reported_effect
from .benchmark import benchmark_trials, run_benchmark
from .robustness import (
    NegativeControlResult,
    SensitivityReport,
    negative_control_check,
    sensitivity_report,
)
from .calibration import (
    CONTROL_TRIALS,
    CalibrationReport,
    CalibrationResult,
    ControlTrial,
    ExpectedEffect,
    calibrate,
    observed_direction,
)
from .variants import (
    SubgroupEstimate,
    SubgroupReport,
    run_estimand_variants,
    run_subgroups,
)

__all__ = [
    "run_tte",
    "add_treatment_indicator",
    "TTEResult",
    "EffectMeasure",
    "ComparisonResult",
    "parse_reported_effect",
    "compare_trial",
    "run_benchmark",
    "benchmark_trials",
    "sensitivity_report",
    "SensitivityReport",
    "negative_control_check",
    "NegativeControlResult",
    "calibrate",
    "CalibrationReport",
    "CalibrationResult",
    "ControlTrial",
    "ExpectedEffect",
    "observed_direction",
    "CONTROL_TRIALS",
    "run_estimand_variants",
    "run_subgroups",
    "SubgroupReport",
    "SubgroupEstimate",
]
