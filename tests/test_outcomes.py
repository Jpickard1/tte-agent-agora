"""#146 outcome selection: emulate the MEASURABLE outcome (prefer binary mortality),
not blindly spec.outcomes[0] (which is often a non-measurable endpoint -> KeyError ->
trial drops). Pure (no pandas/analysis extra)."""

from tteEngine.analysis.outcomes import (
    is_mortality_outcome,
    outcome_column,
    select_measurable_outcome,
)
from tteEngine.contracts.trial_spec import OutcomeSpec, TargetTrialSpec


def _spec(*outcomes):
    return TargetTrialSpec(nct_id="NCT1", outcomes=list(outcomes))


def _o(name, kind="binary", horizon=None):
    return OutcomeSpec(name=name, kind=kind, horizon_hours=horizon)


def test_outcome_column_matches_cohort_builder():
    assert outcome_column("Mortality at 30 Days") == "outcome_Mortality_at_30_Days"


def test_is_mortality_outcome():
    assert is_mortality_outcome("ICU Mortality")
    assert is_mortality_outcome("28-day all-cause death")
    assert not is_mortality_outcome("Quality of Life (EQ-5D)")


def test_picks_measurable_over_first_listed():
    # EQ-5D listed first but NOT materializable; mortality is -> pick mortality.
    spec = _spec(_o("Quality of Life (EQ-5D-5L)"), _o("Mortality at 30 Days", horizon=720))
    cols = ["T", "outcome_Mortality_at_30_Days"]   # only mortality materialized
    chosen = select_measurable_outcome(spec, cols)
    assert chosen is not None and chosen.name == "Mortality at 30 Days"


def test_prefers_mortality_when_both_measurable():
    spec = _spec(_o("Length of ICU Stay"), _o("ICU Mortality"))
    cols = ["outcome_Length_of_ICU_Stay", "outcome_ICU_Mortality"]
    assert select_measurable_outcome(spec, cols).name == "ICU Mortality"


def test_none_when_no_outcome_measurable():
    spec = _spec(_o("Quality of Life (EQ-5D-5L)"), _o("Cognitive Score"))
    assert select_measurable_outcome(spec, ["T", "age"]) is None


def test_falls_back_to_any_measurable_non_mortality():
    spec = _spec(_o("Length of Hospital Stay"))
    cols = ["outcome_Length_of_Hospital_Stay"]
    assert select_measurable_outcome(spec, cols).name == "Length of Hospital Stay"


# --- compare-alignment: pick the REPORTED outcome matching the emulated one ---

def _two_outcome_study():
    """First-listed PRIMARY = non-mortality (RR 1.0); secondary = mortality (RR 0.8)."""
    def om(typ, title, te, tn, ce, cn):
        return {"type": typ, "title": title, "paramType": "COUNT_OF_PARTICIPANTS",
                "groups": [{"id": "OG0", "title": "Drug"}, {"id": "OG1", "title": "Placebo"}],
                "denoms": [{"counts": [{"groupId": "OG0", "value": str(tn)},
                                       {"groupId": "OG1", "value": str(cn)}]}],
                "classes": [{"categories": [{"measurements": [
                    {"groupId": "OG0", "value": str(te)}, {"groupId": "OG1", "value": str(ce)}]}]}]}
    return {"protocolSection": {"identificationModule": {"nctId": "NCT9"}},
            "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": [
                om("PRIMARY", "ICU readmission", 40, 400, 40, 400),       # RR 1.0
                om("SECONDARY", "28-day mortality", 120, 400, 150, 400),  # RR 0.8
            ]}}}


def test_parse_reported_effect_aligns_to_emulated_outcome():
    from tteEngine.analysis import parse_reported_effect
    # default -> primary (ICU readmission, RR ~1.0)
    base = parse_reported_effect(_two_outcome_study(), treatment_hint="drug")
    assert "readmission" in base["title"].lower()
    # aligned to the emulated mortality outcome -> picks 28-day mortality (RR 0.8)
    aligned = parse_reported_effect(_two_outcome_study(), treatment_hint="drug",
                                    outcome_name="ICU Mortality")
    assert "mortality" in aligned["title"].lower()
    assert abs(aligned["effect"]["risk_ratio"] - 0.8) < 1e-9


def test_compare_trial_uses_emulated_outcome_for_alignment():
    from tteEngine.analysis import compare_trial
    from tteEngine.contracts.results import EffectMeasure, TTEResult
    em = TTEResult(nct_id="NCT9", dataset="MIMIC-IV", method="iptw", measure=EffectMeasure.OR,
                   estimate=0.82, ci_low=0.70, ci_high=0.95, extra={"outcome": "ICU Mortality"})
    r = compare_trial(_two_outcome_study(), em, treatment_hint="drug")
    # aligned to mortality (RR 0.8) -> observed ~0.8, concordant (both protective)
    assert r.observed_estimate is not None and abs(r.observed_estimate - 0.8) < 1e-9
