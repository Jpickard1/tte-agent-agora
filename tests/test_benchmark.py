"""Tests for #11 — emulated-vs-observed comparison + batch benchmark (probe).
Pure: builds contracts.TTEResult directly + a synthetic ctgov results fixture
(no analysis extra needed)."""

from tteEngine.analysis import benchmark_trials, compare_trial, parse_reported_effect, run_benchmark
from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult


def _study(nct="NCT001", treated_events=120, treated_n=400,
           control_events=150, control_n=400) -> dict:
    """Study with a PRIMARY count outcome: Hydrocortisone vs Placebo mortality."""
    return {
        "protocolSection": {"identificationModule": {"nctId": nct}},
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [{
            "type": "PRIMARY", "title": "28-day mortality", "timeFrame": "28 days",
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


def _tte(estimate, ci_low, ci_high, measure=EffectMeasure.OR, nct="NCT001", dataset="MIMIC-IV"):
    return TTEResult(nct_id=nct, dataset=dataset, method="iptw", measure=measure,
                     estimate=estimate, ci_low=ci_low, ci_high=ci_high,
                     n_treated=200, n_control=200)


def test_parse_reported_effect():
    eff = parse_reported_effect(_study(), treatment_hint="hydrocortisone")["effect"]
    assert eff["treated_arm"] == "Hydrocortisone" and eff["control_arm"] == "Placebo"
    assert abs(eff["p_treated"] - 30.0) < 1e-9 and abs(eff["p_control"] - 37.5) < 1e-9
    assert eff["direction"] == "lower" and abs(eff["risk_ratio"] - 0.8) < 1e-9


def test_compare_trial_concordant():
    r = compare_trial(_study(), _tte(0.85, 0.74, 0.97), treatment_hint="hydrocortisone")
    assert isinstance(r, ComparisonResult)
    assert r.agreement == Agreement.CONCORDANT          # both protective
    assert r.observed_measure == EffectMeasure.RR and r.nct_id == "NCT001"


def test_compare_trial_discordant():
    r = compare_trial(_study(), _tte(1.4, 1.1, 1.8), treatment_hint="hydrocortisone")
    assert r.agreement == Agreement.DISCORDANT          # emulation says harm, trial says benefit


def test_compare_trial_inconclusive_when_ci_spans_null():
    r = compare_trial(_study(), _tte(0.95, 0.80, 1.15), treatment_hint="hydrocortisone")
    assert r.agreement == Agreement.INCONCLUSIVE        # emulated CI spans 1


def test_run_benchmark_streams_and_aggregates():
    rows, summary = benchmark_trials([
        (_study("NCT001"), _tte(0.85, 0.74, 0.97, nct="NCT001", dataset="MIMIC-IV"), "hydrocortisone", "MIMIC-IV"),
        (_study("NCT002"), _tte(1.4, 1.1, 1.8, nct="NCT002", dataset="eICU"), "hydrocortisone", "eICU"),
        (_study("NCT003"), _tte(0.95, 0.80, 1.15, nct="NCT003", dataset="MIMIC-IV"), "hydrocortisone", "MIMIC-IV"),
    ])
    assert summary["n"] == 3
    assert summary["by_agreement"]["concordant"] == 1
    assert summary["by_agreement"]["discordant"] == 1
    assert summary["n_comparable"] == 2 and summary["concordance_rate"] == 0.5
    assert summary["by_dataset"]["MIMIC-IV"]["concordant"] == 1


def test_run_benchmark_accepts_generator():
    # streaming over a generator (scales to a large corpus without materializing)
    gen = (compare_trial(_study(f"NCT{i}"), _tte(0.85, 0.74, 0.97, nct=f"NCT{i}"),
                         treatment_hint="hydrocortisone") for i in range(50))
    summary = run_benchmark(gen)
    assert summary["n"] == 50 and summary["concordance_rate"] == 1.0
