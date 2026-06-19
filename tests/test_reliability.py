"""Tests for #41 — corpus-level calibration (emulated-vs-observed reliability).
Pure: synthetic ComparisonResults with known calibration (no analysis extra)."""

from tteEngine.analysis import corpus_calibration
from tteEngine.analysis.reliability import CalibrationCurve
from tteEngine.contracts.results import ComparisonResult, EffectMeasure, TTEResult


def _comp(nct, em_est, obs, em_lo=None, em_hi=None) -> ComparisonResult:
    return ComparisonResult(
        nct_id=nct, dataset="MIMIC-IV", agreement="concordant",
        observed_estimate=obs, observed_measure=EffectMeasure.RR,
        emulated=TTEResult(nct_id=nct, dataset="MIMIC-IV", method="iptw",
                           measure=EffectMeasure.OR, estimate=em_est,
                           ci_low=em_lo if em_lo is not None else em_est * 0.6,
                           ci_high=em_hi if em_hi is not None else em_est * 1.7),
    )


def test_perfect_calibration():
    ests = [0.5, 0.7, 1.0, 1.4, 2.0]
    cur = corpus_calibration([_comp(f"N{i}", e, e) for i, e in enumerate(ests)])
    assert isinstance(cur, CalibrationCurve) and cur.n == 5
    assert abs(cur.slope - 1.0) < 1e-6 and abs(cur.intercept) < 1e-6
    assert abs(cur.pearson_r - 1.0) < 1e-6 and cur.coverage == 1.0 and cur.rmse < 1e-9


def test_slope_detects_miscalibration():
    ests = [0.5, 0.8, 1.25, 2.0]
    cur = corpus_calibration([_comp(f"N{i}", e, e ** 1.5) for i, e in enumerate(ests)])
    assert cur.slope > 1.3   # observed = emulated^1.5 -> slope ~1.5


def test_intercept_detects_bias():
    ests = [0.5, 0.8, 1.25, 2.0]
    cur = corpus_calibration([_comp(f"N{i}", e, e * 1.5) for i, e in enumerate(ests)])
    assert abs(cur.slope - 1.0) < 0.05 and cur.intercept > 0.3   # log(1.5) ~ 0.405


def test_coverage_and_skips_non_ratio_or_missing():
    rows = [
        _comp("A", 0.8, 0.85, em_lo=0.6, em_hi=1.0),  # ratio + observed within CI
        ComparisonResult(nct_id="B", dataset="d", agreement="concordant",
                         observed_estimate=0.02, observed_measure=EffectMeasure.RD,
                         emulated=TTEResult(nct_id="B", dataset="d", method="iptw",
                                            measure=EffectMeasure.RD, estimate=0.01,
                                            ci_low=-0.02, ci_high=0.04)),  # RD -> skipped
        ComparisonResult(nct_id="C", dataset="d", agreement="inconclusive",
                         emulated=TTEResult(nct_id="C", dataset="d", method="iptw",
                                            measure=EffectMeasure.OR, estimate=0.9,
                                            ci_low=0.7, ci_high=1.1)),  # no observed -> skipped
    ]
    cur = corpus_calibration(rows)
    assert cur.n == 1 and cur.coverage == 1.0
