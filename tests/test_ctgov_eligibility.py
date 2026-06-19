"""#28 rigorous eligibility: ctgov free-text inclusion/exclusion -> executable
EligibilityCriterion predicates. Pure stdlib (no pandas) -> runs in CI [dev]."""

from tteEngine import vocab
from tteEngine.contracts import Comparator, EventType, TargetTrialSpec
from tteEngine.ctgov.eligibility import (
    enrich_spec_eligibility, parse_eligibility, parse_eligibility_text,
)

SEPSIS_TEXT = """
Inclusion Criteria:
- Adults at least 18 years of age
- Suspected or confirmed infection
- Septic shock requiring vasopressors
- Serum lactate > 2 mmol/L
- Mechanically ventilated

Exclusion Criteria:
* Pregnancy
* Receiving renal replacement therapy
* Enrolled in another interventional study
"""


def _by_concept(parse):
    return {c.concept: c for c in parse.criteria}


def test_sections_split_inclusion_vs_exclusion():
    p = parse_eligibility_text(SEPSIS_TEXT)
    incl = {c.concept for c in p.criteria if c.include}
    excl = {c.concept for c in p.criteria if not c.include}
    assert "sepsis" in incl and "infection" in incl
    assert "pregnancy" in excl and "dialysis" in excl


def test_numeric_threshold_predicate():
    p = parse_eligibility_text(SEPSIS_TEXT)
    lac = _by_concept(p)["lactate"]
    assert lac.event_type == EventType.LAB
    assert lac.comparator == Comparator.GT and lac.value == 2.0 and lac.unit == "mmol/L"


def test_age_demographic_numeric():
    p = parse_eligibility_text("Inclusion Criteria:\n- Age >= 18 years")
    age = _by_concept(p)["age"]
    assert age.event_type == EventType.DEMOGRAPHIC
    assert age.comparator == Comparator.GE and age.value == 18.0


def test_diagnosis_concept_is_presence_not_numeric():
    # 'septic shock requiring vasopressors' is a presence predicate, NOT numeric,
    # even though the trial elsewhere has an age number.
    p = parse_eligibility_text("Inclusion Criteria:\n- Septic shock requiring vasopressors")
    sep = _by_concept(p)["sepsis"]
    assert sep.comparator == Comparator.EXISTS and sep.value is None


def test_exclusion_flagged_include_false():
    p = parse_eligibility_text(SEPSIS_TEXT)
    assert _by_concept(p)["pregnancy"].include is False


def test_unparsed_bullets_recorded_not_dropped():
    p = parse_eligibility_text(SEPSIS_TEXT)
    # 'Enrolled in another interventional study' has no clinical concept -> unparsed
    assert any("another interventional study" in u.lower() for u in p.unparsed)
    # nothing vanishes: parsed + unparsed == total bullets
    assert len(p.criteria) + len(p.unparsed) == p.n_bullets
    assert 0.0 < p.coverage <= 1.0


def test_number_taken_after_operator():
    # 'PaO2/FiO2 ... less than 300' must not grab the '2' from PaO2 (concept unknown
    # here, but creatinine exercises the after-operator search with a leading number)
    p = parse_eligibility_text("Inclusion:\n- 2 prior episodes, serum creatinine > 1.5 mg/dL")
    cr = _by_concept(p)["creatinine"]
    assert cr.value == 1.5 and cr.comparator == Comparator.GT


def test_window_stamped_on_lab_predicates_only():
    p = parse_eligibility_text(SEPSIS_TEXT, window_hours=(-24.0, 24.0))
    assert _by_concept(p)["lactate"].window_hours == (-24.0, 24.0)   # LAB -> windowed
    assert _by_concept(p)["sepsis"].window_hours is None             # presence -> no window


def test_empty_text_is_empty_parse():
    p = parse_eligibility_text(None)
    assert p.criteria == [] and p.n_bullets == 0 and p.coverage == 0.0


def test_no_headers_defaults_to_inclusion():
    p = parse_eligibility_text("- Sepsis\n- Lactate > 4 mmol/L")
    assert all(c.include for c in p.criteria) and len(p.criteria) == 2


def test_classify_fallback_recognizes_raw_codes():
    # a raw ICD code in the text resolves via vocab.classify (A41 -> sepsis)
    p = parse_eligibility_text("Inclusion Criteria:\n- Documented A41 at admission",
                               classify=vocab.classify)
    assert _by_concept(p)["sepsis"].comparator == Comparator.EXISTS


def test_parse_eligibility_from_study_dict():
    study = {"protocolSection": {"eligibilityModule": {"eligibilityCriteria": SEPSIS_TEXT}}}
    p = parse_eligibility(study)
    assert "sepsis" in {c.concept for c in p.criteria}


def test_enrich_spec_merges_and_dedups():
    spec = TargetTrialSpec(nct_id="NCT-X", condition="Sepsis")  # demographics-only / empty
    study = {"protocolSection": {"eligibilityModule": {"eligibilityCriteria": SEPSIS_TEXT}}}
    enrich_spec_eligibility(spec, study)
    concepts = {c.concept for c in spec.eligibility}
    assert {"sepsis", "lactate", "pregnancy"} <= concepts
    # idempotent: a second enrich adds nothing (dedup by predicate identity)
    n = len(spec.eligibility)
    enrich_spec_eligibility(spec, study)
    assert len(spec.eligibility) == n


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
