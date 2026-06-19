"""Tests for #37 — E-values + negative-control sensitivity analyses (probe).
Needs the analysis extra (run_tte) — skip cleanly when absent."""

import numpy as np
import pandas as pd
import pytest

from tteEngine.analysis import SensitivityReport, run_tte, sensitivity_report

pytest.importorskip("statsmodels")
pytest.importorskip("lifelines")


def _cohort(n=1000, seed=0) -> pd.DataFrame:
    """Confounded cohort: sicker (high sofa) patients are treated. `death` is the
    true outcome (protective treatment effect). `neg` is a NEGATIVE CONTROL —
    driven by sofa only, NOT by treatment — so it should be null after adjusting
    for sofa, but spuriously associated with T when unadjusted."""
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, n)
    sofa = rng.normal(7, 3, n)
    T = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.12 * (sofa - 7)))).astype(int)
    death = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + 0.18 * (sofa - 7) - 0.4 * T)))).astype(int)
    neg = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-1 + 0.30 * (sofa - 7))))).astype(int)
    return pd.DataFrame({"T": T, "age": age, "sofa": sofa, "death": death, "neg": neg,
                         "TRAJECTORY_ID": np.arange(n)})


def test_sensitivity_report_iptw_passes():
    df = _cohort()
    primary = run_tte(df, outcome_col="death", covariates=["age", "sofa"], adjustment="iptw")
    rep = sensitivity_report(primary, df, negative_outcome_cols=["neg"],
                             covariates=["age", "sofa"], adjustment="iptw")
    assert isinstance(rep, SensitivityReport)
    assert rep.e_value_point is not None and rep.e_value_note          # E-value reported
    assert rep.n_controls == 1
    assert rep.negative_controls[0].flagged is False                   # adjusted -> null
    assert rep.passed is True


def test_unadjusted_flags_confounded_negative_control():
    df = _cohort()
    primary = run_tte(df, outcome_col="death", covariates=["sofa"], adjustment="unadjusted")
    rep = sensitivity_report(primary, df, negative_outcome_cols=["neg"],
                             covariates=["sofa"], adjustment="unadjusted")
    # unadjusted: T tracks sofa -> spurious effect on the negative control -> flagged
    assert rep.negative_controls[0].flagged is True
    assert rep.n_controls_flagged == 1 and rep.passed is False


def test_no_controls_supplied_note():
    df = _cohort()
    primary = run_tte(df, outcome_col="death", covariates=["age", "sofa"], adjustment="iptw")
    rep = sensitivity_report(primary, df, negative_outcome_cols=[],
                             covariates=["age", "sofa"], adjustment="iptw")
    assert rep.n_controls == 0 and "negative-control" in rep.note
