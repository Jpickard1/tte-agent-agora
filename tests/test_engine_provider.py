"""Test the engine provider (#10 -> Pipeline #12 bridge): it returns the unified
contracts.TTEResult and adjusts for confounders. Needs the analysis extra.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("lifelines")
pytest.importorskip("statsmodels")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import sepsis_vignette as vig  # noqa: E402

from tteEngine.common_format import Aggregation, FeatureSpec  # noqa: E402
from tteEngine.contracts.events import EventType  # noqa: E402
from tteEngine.contracts.results import TTEResult  # noqa: E402
from tteEngine.orchestration.engine_provider import make_engine_provider  # noqa: E402


def test_provider_returns_unified_contract_type():
    stream = vig.confounded_stream(2)
    spec = vig.demo_spec()
    from tteEngine.cohort import build_cohort

    cohort = build_cohort(stream, spec, dataset="MIMIC-IV")
    cov = [FeatureSpec(name="lactate_max", event_type=EventType.LAB, event_name="lactate",
                       agg=Aggregation.MAX, window_hours=(-24.0, 24.0))]
    provider = make_engine_provider(cov, adjustment="iptw")
    result = provider(stream, cohort, spec)
    # the single public seam, regardless of what run_tte returns internally
    assert isinstance(result, TTEResult)
    assert result.nct_id == spec.nct_id and result.dataset == "MIMIC-IV"
    assert result.estimate < 1.0          # adjusted benefit
    assert result.n_treated > 0 and result.n_control > 0
