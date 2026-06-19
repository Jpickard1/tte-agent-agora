"""Tests for the TTE engine entrypoint (#10, probe).

The ported engine (engine.py) needs the `analysis` extra (lifelines/statsmodels);
skip cleanly when absent. run_tte itself + TTEResult import without them.
"""

import numpy as np
import pandas as pd
import pytest

from tteEngine.analysis import TTEResult, add_treatment_indicator, run_tte

pytest.importorskip("statsmodels")
pytest.importorskip("lifelines")


def _confounded_cohort(n=600, seed=0) -> pd.DataFrame:
    """Synthetic confounded ICU cohort: sicker patients get treated AND die more,
    so the crude effect is biased toward harm; adjustment should pull it back."""
    rng = np.random.default_rng(seed)
    age = rng.normal(65, 12, n)
    sofa = rng.normal(7, 3, n)
    ps = 1 / (1 + np.exp(-(0.05 * (sofa - 7) + 0.01 * (age - 65))))
    T = (rng.uniform(size=n) < ps).astype(int)
    logit = -1.0 + 0.18 * (sofa - 7) - 0.25 * T  # true protective effect of T
    death = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {"T": T, "age": age, "sofa": sofa, "death28": death, "TRAJECTORY_ID": np.arange(n)}
    )


def test_run_tte_binary_logistic_iptw():
    r = run_tte(_confounded_cohort(), outcome_col="death28", covariates=["age", "sofa"],
                adjustment="iptw", binary_test="logistic", label="28-day mortality")
    assert isinstance(r, TTEResult) and r.ok
    assert r.effect_measure == "Odds Ratio"
    assert r.point_estimate is not None and r.ci_low <= r.point_estimate <= r.ci_high
    assert r.n_treated + r.n_control == r.n_analyzed > 0
    assert r.e_value_point is not None
    assert {b.variable for b in r.balance} == {"age", "sofa"}


def test_adjustment_moves_estimate_toward_truth():
    # crude (unadjusted) is biased by confounding; IPTW should move toward the
    # true protective OR (<1) — i.e. adjusted point estimate is lower than crude.
    df = _confounded_cohort()
    crude = run_tte(df, outcome_col="death28", covariates=["age", "sofa"],
                    adjustment="unadjusted", binary_test="logistic")
    adj = run_tte(df, outcome_col="death28", covariates=["age", "sofa"],
                  adjustment="iptw", binary_test="logistic")
    assert crude.ok and adj.ok
    assert adj.point_estimate < crude.point_estimate


def test_add_treatment_indicator():
    df = pd.DataFrame({"group": ["hydrocortisone", "placebo", "hydrocortisone"]})
    out = add_treatment_indicator(df, group_col="group", treated_value="hydrocortisone")
    assert out["T"].tolist() == [1, 0, 1]


def test_run_tte_survival_cox():
    rng = np.random.default_rng(1)
    n = 500
    T = rng.integers(0, 2, n)
    sofa = rng.normal(7, 3, n)
    time = rng.exponential(20 + 5 * T - 0.5 * (sofa - 7), n).clip(0.1, 28)
    event = (time < 28).astype(int)
    df = pd.DataFrame({"T": T, "sofa": sofa, "t_time": time, "t_event": event,
                       "TRAJECTORY_ID": np.arange(n)})
    r = run_tte(df, outcome_col="t_event", time_col="t_time", outcome_kind="survival",
                covariates=["sofa"], adjustment="iptw", survival_test="cox")
    assert r.ok and r.effect_measure == "Hazard Ratio"
    assert r.point_estimate is not None
