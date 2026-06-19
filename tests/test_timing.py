"""#31 cross-dataset clock + window harmonization: one TimingConfig drives every
adapter's window + a common timestamp precision. Helper/contract tests are pure;
the adapter-integration + harmonize tests guard on pandas."""

import pytest

from tteEngine import timing as T
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan
from tteEngine.contracts.events import EventType
from tteEngine.contracts.timing import ClockReference, TimePrecision, TimingConfig


def test_timing_config_defaults_and_window_validator():
    tc = TimingConfig()
    assert tc.clock == ClockReference.ICU_ADMISSION and tc.precision == TimePrecision.MINUTE
    assert tc.extraction_window_hours == (-48.0, 24.0)
    with pytest.raises(ValueError):
        TimingConfig(extraction_window_hours=(10.0, -10.0))   # lo > hi rejected


def test_effective_window_prefers_timing_else_legacy():
    legacy = ExtractionPlan(nct_id="X", window_hours=(-10.0, 5.0))
    assert T.effective_window(legacy) == (-10.0, 5.0)         # back-compat: no timing
    timed = ExtractionPlan(nct_id="X", window_hours=(-10.0, 5.0),
                           timing=TimingConfig(extraction_window_hours=(-24.0, 12.0)))
    assert T.effective_window(timed) == (-24.0, 12.0)         # timing wins


def test_to_time_zero_rule_bridges_clock_and_grace():
    tc = TimingConfig(clock=ClockReference.HOSPITAL_ADMISSION, grace_window_hours=12.0)
    tzr = T.to_time_zero_rule(tc)
    assert tzr.anchor == "hospital_admission" and tzr.grace_window_hours == 12.0


def test_precision_warning_when_finer_than_native():
    tc = TimingConfig(precision=TimePrecision.SECOND)
    assert T.precision_warnings(tc, "eICU-CRD")               # second finer than eICU minute -> warn
    assert T.precision_warnings(TimingConfig(precision=TimePrecision.HOUR), "eICU-CRD") == []


def test_harmonize_floors_timestamp_precision():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"TIMESTAMP": [pd.Timestamp("2024-01-01 03:25:30.5", tz="UTC"),
                                     pd.Timestamp("2024-01-01 03:55:00", tz="UTC")]})
    out = T.harmonize_timestamps(df, TimingConfig(precision=TimePrecision.HOUR))
    assert (out["TIMESTAMP"] == pd.Timestamp("2024-01-01 03:00", tz="UTC")).all()
    # None -> untouched (back-compat)
    assert T.harmonize_timestamps(df, None) is df


def test_timing_drives_eicu_window_and_precision():
    pd = pytest.importorskip("pandas")
    from tteEngine.adapters import eicu

    tables = {
        "patient": pd.DataFrame({"patientunitstayid": [10]}),
        "diagnosis": pd.DataFrame({"patientunitstayid": [10], "icd9code": ["A41"],
                                   "diagnosisstring": ["sepsis"], "diagnosisoffset": [0]}),
        "lab": pd.DataFrame({"patientunitstayid": [10, 10], "labname": ["lactate", "lactate"],
                             "labresult": [2.0, 9.0], "labresultoffset": [30, 90]}),  # 0.5h in, 1.5h out
    }
    plan = ExtractionPlan(
        nct_id="X", cohort_filter_concepts=["A41"],
        concepts=[ConceptRequest(concept="A41", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="lactate", event_type=EventType.LAB, role="covariate")],
        timing=TimingConfig(extraction_window_hours=(-1.0, 1.0), precision=TimePrecision.HOUR),
    )
    df = eicu.extract(plan, tables)
    labs = df[df["EVENT_TYPE"] == "lab"]
    assert list(labs["EVENT_VALUE"]) == ["2.0"]               # +90m dropped by the 1h timing window
    # precision=HOUR floored every timestamp to the hour (eICU EPOCH = 2000-01-01)
    assert (df["TIMESTAMP"] == eicu.EPOCH).all()


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
