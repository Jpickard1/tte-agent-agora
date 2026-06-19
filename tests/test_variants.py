"""Tests for #38 — estimand variants + prespecified subgroups (probe). Needs the
analysis extra (run_tte) — skip cleanly when absent."""

import numpy as np
import pandas as pd
import pytest

from tteEngine.analysis import run_estimand_variants, run_subgroups
from tteEngine.analysis.variants import SubgroupReport

pytest.importorskip("statsmodels")
pytest.importorskip("lifelines")


def _cohort_with_effect_modification(n=1200, seed=0) -> pd.DataFrame:
    """Treatment is PROTECTIVE in sex=0 and HARMFUL in sex=1 (true effect
    modification) -> subgroup estimates should diverge."""
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, n)
    sofa = rng.normal(7, 3, n)
    T = rng.integers(0, 2, n)
    # treatment coefficient flips sign by subgroup
    beta_T = np.where(sex == 0, -1.0, 1.0)
    logit = -0.5 + 0.1 * (sofa - 7) + beta_T * T
    death = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    adhere = (rng.uniform(size=n) < 0.8).astype(int)  # 80% adherence flag
    return pd.DataFrame({"T": T, "sofa": sofa, "sex": sex, "death": death,
                         "adhere": adhere, "TRAJECTORY_ID": np.arange(n)})


def test_estimand_variants_itt_and_per_protocol():
    df = _cohort_with_effect_modification()
    out = run_estimand_variants(df, outcome_col="death", covariates=["sofa"],
                                adherence_col="adhere", adjustment="iptw")
    assert set(out) == {"itt", "per_protocol"}
    assert out["itt"].extra["ok"] and out["per_protocol"].extra["ok"]
    # per-protocol restricts to adherent rows -> fewer analyzed
    assert out["per_protocol"].extra["n_analyzed"] < out["itt"].extra["n_analyzed"]


def test_subgroups_detect_effect_modification():
    df = _cohort_with_effect_modification()
    rep = run_subgroups(df, subgroup_col="sex", outcome_col="death", covariates=["sofa"],
                        adjustment="iptw")
    assert isinstance(rep, SubgroupReport)
    assert {s.subgroup for s in rep.subgroups} == {"sex=0", "sex=1"}
    assert rep.heterogeneity is True   # protective in one arm, harmful in the other


def test_subgroups_no_modification_when_uniform():
    rng = np.random.default_rng(1)
    n = 1200
    sex = rng.integers(0, 2, n)
    T = rng.integers(0, 2, n)
    sofa = rng.normal(7, 3, n)
    death = (rng.uniform(size=n) < 1 / (1 + np.exp(-(-0.5 + 0.1 * (sofa - 7) - 0.4 * T)))).astype(int)
    df = pd.DataFrame({"T": T, "sofa": sofa, "sex": sex, "death": death,
                       "TRAJECTORY_ID": np.arange(n)})
    rep = run_subgroups(df, subgroup_col="sex", outcome_col="death", covariates=["sofa"],
                        adjustment="iptw")
    assert rep.heterogeneity is False  # same effect in both subgroups -> CIs overlap
