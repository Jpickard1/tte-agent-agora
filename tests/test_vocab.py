"""Tests for the vocab/concept-normalization layer (#5)."""

from datetime import datetime, timezone

from tteEngine import vocab
from tteEngine.contracts.events import Event, EventType


def test_resolve_concept_to_codes():
    codes = vocab.resolve("sepsis")
    assert "A41" in codes and "99592" in codes        # ICD-10 + ICD-9 both seeded
    assert vocab.resolve("unknownX") == {"unknownX"}    # fallback to literal


def test_classify_code_to_concept():
    assert vocab.classify("A41") == "sepsis"
    assert vocab.classify("Norepinephrine") == "vasopressor"   # case-insensitive
    assert vocab.classify("hydrocortisone") == "corticosteroid"
    assert vocab.classify("ZZZ999") is None


def test_normalize_numeric_with_unit_in_name():
    val, unit, text = vocab.normalize_value("1.2", "Creatinine (mg/dL)")
    assert val == 1.2 and unit == "mg/dL" and text is None


def test_normalize_json_value():
    val, unit, text = vocab.normalize_value('{"dose": 50, "unit": "mg"}', "hydrocortisone")
    assert val == 50.0 and unit == "mg" and text is None


def test_normalize_categorical_text():
    val, unit, text = vocab.normalize_value("Positive", "Blood Culture")
    assert val is None and text == "Positive"


def test_to_normalized_event_builds_sidecar():
    ev = Event(trajectory_id=1, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
               event_type=EventType.LAB, event_name="Creatinine (mg/dL)", event_value="1.2")
    n = vocab.to_normalized_event(ev)
    assert n.trajectory_id == 1 and n.event_type == EventType.LAB
    assert n.value_num == 1.2 and n.unit == "mg/dL" and n.concept_id is None  # creatinine unmapped (ok)
    # a mapped one:
    ev2 = Event(trajectory_id=1, timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                event_type=EventType.DIAGNOSIS, event_name="A41", event_value="A41")
    assert vocab.to_normalized_event(ev2).concept_id == "sepsis"


def test_register_concept_extends_in_repo():
    vocab.register_concept("aki", {"N17", "N179"})
    assert "N17" in vocab.resolve("aki") and vocab.classify("N179") == "aki"


def run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
