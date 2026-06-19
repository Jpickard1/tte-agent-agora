"""Real-code smoke test (#59 resolver + #71 hook on real adapter-shaped output):
the confounding flip reproduces when the sepsis dx is a raw ICD code ('A41'),
and the resolver is proven load-bearing (identity -> empty cohort). Skips without
pandas; the IPTW flip additionally needs the analysis extra."""

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import realcode_smoke as smoke  # noqa: E402


def _has_analysis():
    try:
        import lifelines  # noqa: F401
        import statsmodels  # noqa: F401
        return True
    except Exception:
        return False


def test_resolver_is_load_bearing_on_raw_codes():
    # the core #59 proof on real-code shape: 'sepsis' eligibility can't match raw
    # 'A41' under identity (empty cohort) but does with vocab.classify.
    stream = smoke.raw_coded_stream(scale=2)
    assert smoke.cohort_n(stream, resolve=None) == 0
    assert smoke.cohort_n(stream, resolve=smoke.vocab.classify) > 0


def test_crude_on_raw_codes_shows_apparent_harm():
    rep = smoke.run_realcode_crude("MIMIC-IV", smoke.raw_coded_stream(2))
    # same confounded cohort as the #13 vignette -> crude RR ~ 1.10 (apparent harm)
    assert rep.emulated.n_treated == 280 and rep.emulated.n_control == 280
    assert rep.emulated.estimate > 1.0


@pytest.mark.skipif(not _has_analysis(), reason="analysis extra (lifelines/statsmodels) not installed")
def test_iptw_flip_reproduces_on_raw_codes():
    stream = smoke.raw_coded_stream(2)
    crude = smoke.run_realcode_crude("MIMIC-IV", stream).emulated
    adj = smoke.run_realcode_adjusted("MIMIC-IV", stream).emulated
    assert crude.estimate > 1.0       # apparent harm before adjustment
    assert adj.estimate < 1.0         # benefit after adjusting for lactate
    assert adj.estimate < crude.estimate


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys as _s
    _s.exit(0 if run() else 1)
