"""#13 end-to-end vignette test: the whole spine runs on synthetic MIMIC + eICU
and produces a sensible emulated-vs-observed comparison. Skips without pandas.
"""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

# make examples/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import sepsis_vignette as vig  # noqa: E402

from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure  # noqa: E402


def test_vignette_runs_end_to_end_on_both_datasets():
    reports = vig.run_vignette()
    assert set(reports) == {"MIMIC-IV", "eICU-CRD"}
    for dataset, rep in reports.items():
        assert isinstance(rep, ComparisonResult)
        assert rep.nct_id == "NCT-SEPSIS-STEROID"
        assert rep.emulated.dataset == dataset
        assert rep.emulated.measure == EffectMeasure.RR


def test_vignette_mimic_numbers():
    rep = vig.run_vignette()["MIMIC-IV"]
    e = rep.emulated
    assert e.n_treated == 40 and e.n_control == 60
    # treated risk 18/40=0.45, control 18/60=0.30 -> crude RR 1.5
    assert e.extra["risk_treated"] == pytest.approx(0.45)
    assert e.extra["risk_control"] == pytest.approx(0.30)
    assert e.estimate == pytest.approx(1.5)


def test_crude_estimate_is_confounded_discordant():
    # the whole point: crude RR shows HARM (>1) while the trial reported BENEFIT
    # (<1) -> discordant, motivating the adjusted engine (#10).
    for rep in vig.run_vignette().values():
        assert rep.emulated.estimate > 1.0
        assert rep.observed_estimate < 1.0
        assert rep.agreement == Agreement.DISCORDANT


def test_eligibility_filters_to_septic_high_lactate():
    # control arm has lactate 3.0 (>2) and sepsis -> all eligible; counts intact
    rep = vig.run_vignette()["eICU-CRD"]
    assert rep.emulated.n_treated == 30 and rep.emulated.n_control == 50
