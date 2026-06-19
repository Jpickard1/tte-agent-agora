"""#139 assignment-audit schema: the seam worker1's matcher + probe's cohort emit
into and the #130 UI renders. Import-light (no pandas/analysis needed)."""

from tteEngine.contracts.audit import (
    LOW_CONFIDENCE,
    ArmAudit,
    AssignmentAudit,
    Confidence,
    EligibilityDecision,
    MatchProvenance,
    dump_audit_jsonl,
    load_audit_jsonl,
)


def test_confidence_tiers_and_low_flag():
    # locked tiers; icd_hierarchy = the dx analogue of rxnorm_code (structured-code,
    # high trust), added for code-based diagnosis matching (#129/#132).
    assert [c.value for c in Confidence] == [
        "rxnorm_code", "icd_hierarchy", "ingredient", "name", "substring"]
    assert Confidence.SUBSTRING in LOW_CONFIDENCE
    assert Confidence.RXNORM_CODE not in LOW_CONFIDENCE
    assert Confidence.ICD_HIERARCHY not in LOW_CONFIDENCE   # dx code-family match is high trust


def _audit():
    return AssignmentAudit(
        nct_id="NCT01", dataset="MIMIC-IV",
        n_screened=100, n_eligible=80, n_enrolled=70, n_excluded_immortal=4, n_unassigned=6,
        arms=[
            ArmAudit(name="hydrocortisone", n=40, defining_codes=["RxNorm:5492"],
                     intervention_concepts=["Drug: Hydrocortisone"],
                     match_method_counts={"rxnorm_code": 38, "substring": 2}),
            ArmAudit(name="control", is_control=True, n=30),
        ],
        eligibility=[
            EligibilityDecision(concept="sepsis", event_type="diagn", result="met"),
            EligibilityDecision(concept="age", event_type="demog", measurable=False,
                                result="skipped_unmeasurable", reason="no demographics emitted"),
        ],
        sample=[MatchProvenance(trajectory_id=1, arm="hydrocortisone",
                                matched_event_name="hydrocortisone na succ.", matched_code="RxNorm:5492",
                                concept="Drug: Hydrocortisone", method=Confidence.RXNORM_CODE, t_rel_hours=2.0)],
        n_low_confidence=2)


def test_match_method_totals_rollup():
    a = _audit()
    assert a.match_method_totals() == {"rxnorm_code": 38, "substring": 2}


def test_skipped_eligibility_is_recorded_honestly():
    a = _audit()
    skipped = [e for e in a.eligibility if e.result == "skipped_unmeasurable"]
    assert len(skipped) == 1 and skipped[0].measurable is False
    assert "no demographics" in skipped[0].reason


def test_audit_jsonl_roundtrip(tmp_path):
    p = tmp_path / "audit.jsonl"
    n = dump_audit_jsonl([_audit(), _audit().model_copy(update={"nct_id": "NCT02"})], p)
    assert n == 2
    back = list(load_audit_jsonl(p))
    assert [a.nct_id for a in back] == ["NCT01", "NCT02"]
    # provenance + low-confidence survive the round-trip (the UI ambers these)
    assert back[0].sample[0].method == Confidence.RXNORM_CODE
    assert back[0].n_low_confidence == 2
    assert back[0].arms[0].defining_codes == ["RxNorm:5492"]
