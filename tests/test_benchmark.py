"""Tests for #11 — emulated-vs-observed comparison + batch benchmark (probe).
Pure (no analysis extra needed): builds TTEResult directly + a synthetic ctgov
results fixture."""

from tteEngine.analysis import TTEResult
from tteEngine.analysis.compare import concordance, parse_reported_effect, compare_trial
from tteEngine.analysis.benchmark import benchmark_trials, run_benchmark


def _study(nct="NCT001", treated_events=120, treated_n=400,
           control_events=150, control_n=400) -> dict:
    """A study with a PRIMARY count outcome: Hydrocortisone vs Placebo mortality."""
    return {
        "protocolSection": {"identificationModule": {"nctId": nct}},
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [{
            "type": "PRIMARY",
            "title": "28-day mortality",
            "timeFrame": "28 days",
            "paramType": "COUNT_OF_PARTICIPANTS",
            "groups": [{"id": "OG0", "title": "Hydrocortisone"},
                       {"id": "OG1", "title": "Placebo"}],
            "denoms": [{"units": "Participants",
                        "counts": [{"groupId": "OG0", "value": str(treated_n)},
                                   {"groupId": "OG1", "value": str(control_n)}]}],
            "classes": [{"categories": [{"measurements": [
                {"groupId": "OG0", "value": str(treated_events)},
                {"groupId": "OG1", "value": str(control_events)}]}]}],
        }]}},
    }


def _tte(point, ci_low, ci_high, measure="Odds Ratio") -> TTEResult:
    return TTEResult(ok=True, outcome="28-day mortality", effect_measure=measure,
                     point_estimate=point, ci_low=ci_low, ci_high=ci_high,
                     n_analyzed=400, n_treated=200, n_control=200)


def test_parse_reported_effect():
    eff = parse_reported_effect(_study(), treatment_hint="hydrocortisone")["effect"]
    assert eff["treated_arm"] == "Hydrocortisone" and eff["control_arm"] == "Placebo"
    assert abs(eff["p_treated"] - 30.0) < 1e-9 and abs(eff["p_control"] - 37.5) < 1e-9
    assert eff["direction"] == "lower"  # treated rate < control
    assert abs(eff["risk_ratio"] - 0.8) < 1e-9


def test_concordant_both_protective():
    eff = parse_reported_effect(_study(), treatment_hint="hydrocortisone")["effect"]
    v = concordance(eff, _tte(0.85, 0.74, 0.97))  # OR<1, CI excludes 1
    assert v["verdict"] == "concordant"


def test_discordant_directions_disagree():
    eff = parse_reported_effect(_study(), treatment_hint="hydrocortisone")["effect"]
    v = concordance(eff, _tte(1.4, 1.1, 1.8))  # emulation says harm, trial says benefit
    assert v["verdict"] == "discordant"


def test_inconclusive_when_emulated_ci_spans_one():
    eff = parse_reported_effect(_study(), treatment_hint="hydrocortisone")["effect"]
    v = concordance(eff, _tte(0.95, 0.80, 1.15))  # CI spans 1
    assert v["verdict"] == "inconclusive"


def test_compare_trial_row():
    row = compare_trial(_study(), _tte(0.85, 0.74, 0.97),
                        treatment_hint="hydrocortisone", dataset="MIMIC-IV")
    assert row["nct_id"] == "NCT001" and row["dataset"] == "MIMIC-IV"
    assert row["verdict"] == "concordant"


def test_run_benchmark_aggregates():
    rows, summary = benchmark_trials([
        (_study("NCT001"), _tte(0.85, 0.74, 0.97), "hydrocortisone", "MIMIC-IV"),
        (_study("NCT002"), _tte(1.4, 1.1, 1.8), "hydrocortisone", "eICU"),
        (_study("NCT003"), _tte(0.95, 0.80, 1.15), "hydrocortisone", "MIMIC-IV"),
    ])
    assert summary["n"] == 3
    assert summary["by_verdict"]["concordant"] == 1
    assert summary["by_verdict"]["discordant"] == 1
    assert summary["n_comparable"] == 2
    assert summary["concordance_rate"] == 0.5
