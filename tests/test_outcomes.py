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
