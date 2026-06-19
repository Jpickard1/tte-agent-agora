"""#103 MGB targeted-extraction codegen. Pure string generation — no MGB, no
network, no creds. Asserts the generated Snowflake SQL mirrors mgb.extract's
contract (cohort by dx codes, landmark window, per-type concept filter, canonical
5 columns)."""

from tteEngine.adapters import mgb_codegen as G
from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ConceptRequest, ExtractionPlan


def _plan():
    return ExtractionPlan(
        nct_id="NCT-MGB-SEPSIS", dataset="MGB", cohort_filter_concepts=["sepsis"],
        concepts=[ConceptRequest(concept="sepsis", event_type=EventType.DIAGNOSIS, role="eligibility"),
                  ConceptRequest(concept="death", event_type=EventType.OUTCOME, role="outcome"),
                  ConceptRequest(concept="lactate", event_type=EventType.LAB, role="covariate")],
        window_hours=(-48.0, 24.0))


def _resolve(concept):
    return {"sepsis": {"A419", "R6521"}, "lactate": {"Lactate"}, "death": {"death"}}.get(concept, {concept})


def test_script_has_cohort_window_and_canonical_columns():
    sql = G.generate_mgb_script(_plan(), resolve=_resolve, source_table="MGB.EVENTS")
    assert "NCT-MGB-SEPSIS" in sql and "GATED" in sql
    # cohort CTE filters DIAGNOSIS by the resolved codes
    assert "EVENT_TYPE = 'diagn'" in sql and "'A419'" in sql and "'R6521'" in sql
    # landmark window relative to t0 (mirrors extract)
    assert "MIN(e.TIMESTAMP) OVER (PARTITION BY e.TRAJECTORY_ID) AS T0" in sql
    assert "DATEADD('hour', -48.0, T0)" in sql and "DATEADD('hour', 24.0, T0)" in sql
    # canonical 5 columns projected
    assert "SELECT TRAJECTORY_ID, TIMESTAMP, EVENT_TYPE, EVENT_NAME, EVENT_VALUE" in sql
    assert "FROM MGB.EVENTS" in sql


def test_per_type_concept_filter():
    sql = G.generate_mgb_script(_plan(), resolve=_resolve)
    assert "(EVENT_TYPE = 'lab' AND EVENT_NAME IN ('Lactate'))" in sql
    assert "EVENT_TYPE = 'outco'" in sql and "'death'" in sql      # outcome concept kept


def test_no_cohort_filter_selects_all():
    plan = ExtractionPlan(nct_id="N", concepts=[
        ConceptRequest(concept="lactate", event_type=EventType.LAB, role="covariate")])
    sql = G.generate_mgb_script(plan, resolve=_resolve)
    assert "no cohort filter" in sql and "SELECT DISTINCT TRAJECTORY_ID FROM" in sql


def test_in_list_is_sorted_deterministic_and_escapes_quotes():
    assert G._in_list({"b", "a", "a"}) == "'a', 'b'"
    assert G._in_list({"O'Neil"}) == "'O''Neil'"        # SQL-escaped


def test_write_script_roundtrip(tmp_path):
    path = tmp_path / "mgb.sql"
    sql = G.write_mgb_script(_plan(), path, resolve=_resolve)
    assert path.read_text() == sql and sql.strip().endswith(";")


def run():
    import tempfile
    from pathlib import Path
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        if "tmp_path" in t.__code__.co_varnames[: t.__code__.co_argcount]:
            with tempfile.TemporaryDirectory() as d:
                t(Path(d))
        else:
            t()
        print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
