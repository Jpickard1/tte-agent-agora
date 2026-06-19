"""Tests for #39 — positive/negative control-trial calibration (probe). Pure:
a stub emulator returns known TTEResults (no analysis extra / no network)."""

from tteEngine.analysis.calibration import (
    CONTROL_TRIALS,
    CalibrationReport,
    ControlTrial,
    ExpectedEffect,
    calibrate,
    observed_direction,
)
from tteEngine.contracts.results import EffectMeasure, TTEResult


def _tte(est, lo, hi, measure=EffectMeasure.OR):
    return TTEResult(nct_id="x", dataset="d", method="iptw", measure=measure,
                     estimate=est, ci_low=lo, ci_high=hi)


def test_observed_direction():
    assert observed_direction(_tte(0.6, 0.4, 0.9)) == ExpectedEffect.BENEFIT
    assert observed_direction(_tte(1.5, 1.1, 2.0)) == ExpectedEffect.HARM
    assert observed_direction(_tte(0.95, 0.80, 1.20)) == ExpectedEffect.NULL   # CI spans 1
    assert observed_direction(_tte(0.02, -0.05, 0.09, measure=EffectMeasure.RD)) == ExpectedEffect.NULL


def test_calibrate_all_recovered():
    trials = [
        ControlTrial(nct_id="P1", label="benefit", expected=ExpectedEffect.BENEFIT, kind="positive"),
        ControlTrial(nct_id="P2", label="harm", expected=ExpectedEffect.HARM, kind="positive"),
        ControlTrial(nct_id="N1", label="null", expected=ExpectedEffect.NULL, kind="negative"),
    ]
    emu = {"P1": _tte(0.6, 0.4, 0.9), "P2": _tte(1.5, 1.1, 2.0), "N1": _tte(0.98, 0.85, 1.12)}
    rep = calibrate(trials, lambda ct: emu[ct.nct_id])
    assert isinstance(rep, CalibrationReport)
    assert rep.n == 3 and rep.n_recovered == 3
    assert rep.positive_recovery_rate == 1.0 and rep.negative_specificity == 1.0
    assert rep.passed is True


def test_calibrate_flags_miscalibration():
    trials = [
        ControlTrial(nct_id="P1", label="benefit", expected=ExpectedEffect.BENEFIT, kind="positive"),
        ControlTrial(nct_id="N1", label="null", expected=ExpectedEffect.NULL, kind="negative"),
    ]
    emu = {"P1": _tte(0.98, 0.80, 1.20),   # null but expected benefit -> miss
           "N1": _tte(1.5, 1.1, 2.0)}      # non-null but expected null -> spurious
    rep = calibrate(trials, lambda ct: emu[ct.nct_id])
    assert rep.positive_recovery_rate == 0.0 and rep.negative_specificity == 0.0
    assert rep.passed is False
    assert not rep.results[0].recovered and not rep.results[1].recovered


def test_seed_control_suite_loadable():
    assert CONTROL_TRIALS and all(c.kind in ("positive", "negative") for c in CONTROL_TRIALS)
    assert all(isinstance(c.expected, ExpectedEffect) for c in CONTROL_TRIALS)
