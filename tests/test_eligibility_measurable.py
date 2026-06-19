"""#133 eligibility-measurability resolver — tte1's #138 measurable_fn. Pure
(index is a dict) -> runs in CI [dev]; no pandas, no real data."""

from tteEngine import measurability as M
from tteEngine.contracts.events import EventType

# minimal #109-shaped index: this (fake) dataset has sepsis dx + lactate lab only
INDEX = {"categories": {
    "diagnosis": [{"code": "A419", "name": "Sepsis, unspecified organism"},
                  {"code": "99592", "name": "Severe sepsis"}],
    "lab": [{"code": "50813", "name": "Lactate"}],
    "vital": [{"code": "heartrate", "name": "heart_rate"}]}}


def test_heuristic_without_index():
    ok, reason = M.eligibility_measurable("sepsis", EventType.DIAGNOSIS, "MIMIC-IV")
    assert ok is True and reason                      # domain captured -> measurable


def test_data_driven_resolves_to_codes():
    ok, reason = M.eligibility_measurable("Sepsis", EventType.DIAGNOSIS, "MIMIC-IV", index=INDEX)
    assert ok is True and "real" in reason            # resolves to A419/99592


def test_data_driven_no_codes_is_not_measurable():
    # a lab not present in this dataset's catalog -> not assessable (real check)
    ok, reason = M.eligibility_measurable("procalcitonin", EventType.LAB, "MIMIC-IV", index=INDEX)
    assert ok is False and "no lab codes" in reason


def test_unmeasurable_event_type_short_circuits():
    # LOCATION isn't captured by MGB -> unmeasurable regardless of index
    ok, _ = M.eligibility_measurable("icu", EventType.LOCATION, "MGB", index=INDEX)
    assert ok is False


def test_demographics_use_heuristic_even_with_index():
    # DEMOGRAPHIC has no code catalog -> heuristic (not forced unmeasurable by index)
    ok, _ = M.eligibility_measurable("age", EventType.DEMOGRAPHIC, "MIMIC-IV", index=INDEX)
    assert ok is True


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)


def test_pruned_event_types_skips_not_fails():
    # #163: in a lean run, a lab eligibility criterion is PRUNED -> must be reported
    # not-assessable (skipped), NOT enforced (which would empty the cohort).
    from tteEngine.contracts.events import EventType as ET
    ok, reason = M.eligibility_measurable("lactate", ET.LAB, "MIMIC-IV",
                                          pruned_event_types={ET.LAB, ET.MEASUREMENT})
    assert ok is False and "pruned" in reason
    # a dx criterion (not pruned) is still enforced normally
    ok2, _ = M.eligibility_measurable("sepsis", ET.DIAGNOSIS, "MIMIC-IV",
                                      pruned_event_types={ET.LAB, ET.MEASUREMENT})
    assert ok2 is True
