"""Tests for the TTE engine entrypoint (#10, probe).

run_tte returns the canonical contracts.TTEResult (measure enum + estimate; rich
diagnostics in .extra). The ported engine needs the `analysis` extra
(lifelines/statsmodels) — skip cleanly when absent; contracts import without it.
"""

import numpy as np
import pandas as pd
import pytest

from tteEngine.analysis import add_treatment_indicator, run_tte
from tteEngine.contracts.results import EffectMeasure, TTEResult

pytest.importorskip("statsmodels")
pytest.importorskip("lifelines")


def _confounded_cohort(n=600, seed=0) -> pd.DataFrame:
    """Confounded ICU cohort: sicker patients get treated AND die more, so the
    crude effect is biased toward harm; adjustment should pull it back."""
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


def test_run_tte_returns_contracts_result():
    r = run_tte(_confounded_cohort(), outcome_col="death28", covariates=["age", "sofa"],
                adjustment="iptw", binary_test="logistic", label="28-day mortality",
                nct_id="NCT001", dataset="MIMIC-IV")
    assert isinstance(r, TTEResult)             # the canonical contracts type
    assert r.nct_id == "NCT001" and r.dataset == "MIMIC-IV"
    assert r.measure == EffectMeasure.OR
    assert r.estimate is not None and r.ci_low <= r.estimate <= r.ci_high
    assert r.n_treated + r.n_control == r.extra["n_analyzed"] > 0
    assert r.method == "iptw"
    # rich diagnostics ride in .extra
    assert r.extra["ok"] is True and r.extra["e_value_point"] is not None
    assert {b["variable"] for b in r.extra["balance"]} == {"age", "sofa"}


def test_adjustment_moves_estimate_toward_truth():
    df = _confounded_cohort()
    crude = run_tte(df, outcome_col="death28", covariates=["age", "sofa"],
                    adjustment="unadjusted", binary_test="logistic")
    adj = run_tte(df, outcome_col="death28", covariates=["age", "sofa"],
                  adjustment="iptw", binary_test="logistic")
    assert crude.extra["ok"] and adj.extra["ok"]
    assert adj.estimate < crude.estimate  # IPTW corrects confounding-by-indication


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
    assert r.measure == EffectMeasure.HR and r.estimate is not None
