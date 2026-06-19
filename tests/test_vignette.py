"""#13 end-to-end vignette test: the whole spine runs on synthetic MIMIC + eICU
and demonstrates the confounding flip (crude harm -> adjusted benefit).
Skips without pandas; the adjusted arm additionally needs the analysis extra.
"""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import sepsis_vignette as vig  # noqa: E402

from tteEngine.contracts.results import ComparisonResult  # noqa: E402


def _has_analysis():
    try:
        import lifelines  # noqa: F401
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def test_crude_runs_on_both_datasets_and_shows_apparent_harm():
    streams = {"MIMIC-IV": vig.confounded_stream(2), "eICU-CRD": vig.confounded_stream(1)}
    for ds, s in streams.items():
        rep = vig.run_crude(ds, s)
        assert isinstance(rep, ComparisonResult)
        # confounded-by-indication: the crude estimate is biased toward harm (>1)
        assert rep.emulated.estimate > 1.0


def test_crude_mimic_exact_numbers():
    # deterministic stratified construction (scale=2): treated 88 deaths / 280,
    # control 80 / 280 -> crude RR = (88/280)/(80/280) = 1.10
    rep = vig.run_crude("MIMIC-IV", vig.confounded_stream(2))
    assert rep.emulated.n_treated == 280 and rep.emulated.n_control == 280
    assert rep.emulated.estimate == pytest.approx(1.10, abs=0.005)


@pytest.mark.skipif(not _has_analysis(), reason="analysis extra (lifelines/statsmodels) not installed")
def test_iptw_adjustment_flips_to_benefit():
    # the milestone: adjusting for the confounder reverses crude harm -> benefit
    for ds, scale in (("MIMIC-IV", 2), ("eICU-CRD", 1)):
        stream = vig.confounded_stream(scale)
        crude = vig.run_crude(ds, stream).emulated
        adj = vig.run_adjusted(ds, stream).emulated
        assert crude.estimate > 1.0           # apparent harm before adjustment
        assert adj.estimate < 1.0             # benefit after adjusting for lactate
        assert adj.estimate < crude.estimate  # adjustment moves toward benefit
        assert "p_value" in adj.extra         # rich engine diagnostics via .extra


@pytest.mark.skipif(not _has_analysis(), reason="analysis extra not installed")
def test_run_vignette_structure():
    reports = vig.run_vignette()
    assert set(reports) == {"MIMIC-IV", "eICU-CRD"}
    for r in reports.values():
        assert isinstance(r["crude"], ComparisonResult)
        assert isinstance(r["adjusted"], ComparisonResult)
