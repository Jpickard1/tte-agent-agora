"""#95 per-(nct_id, dataset) CONTEXT sidecar: bundles #35/#33/#34/#32, joined to
the corpus by (nct_id, dataset). Pure parts run in CI [dev]; the missingness
block + JSONL round-trip are exercised here too (no heavy deps)."""

import pytest

from tteEngine import context as C
from tteEngine.contracts.context import (
    TrialDatasetContext, dump_context_jsonl, load_context_jsonl,
)
from tteEngine.contracts.events import EventType
from tteEngine.contracts.results import ComparisonResult, EffectMeasure, TTEResult
from tteEngine.contracts.trial_spec import (
    Arm, Comparator, EligibilityCriterion, OutcomeSpec, TargetTrialSpec,
)


def _spec(nct="NCT-SEP"):
    return TargetTrialSpec(
        nct_id=nct, condition="Septic Shock",
        eligibility=[EligibilityCriterion(concept="sepsis", event_type=EventType.DIAGNOSIS),
                     EligibilityCriterion(concept="map", event_type=EventType.MEASUREMENT,
                                          comparator=Comparator.LT, value=65.0)],
        arms=[Arm(name="steroid", intervention_concepts=["hydrocortisone"]),
              Arm(name="control", is_control=True)],
        outcomes=[OutcomeSpec(name="28-day mortality", concept="death")],
    )


def _cr(dataset, est, lo, hi, n_t, n_c):
    return ComparisonResult(
        nct_id="NCT-SEP", dataset=dataset,
        emulated=TTEResult(nct_id="NCT-SEP", dataset=dataset, method="iptw",
                           measure=EffectMeasure.OR, estimate=est, ci_low=lo, ci_high=hi,
                           n_treated=n_t, n_control=n_c))


def test_build_context_has_join_key_and_all_blocks():
    ctx = C.build_context(_spec(), "MIMIC-IV")
    assert ctx.nct_id == "NCT-SEP" and ctx.dataset == "MIMIC-IV"   # the join key
    assert ctx.is_sepsis is True
    assert ctx.emulability and "score" in ctx.emulability          # #35
    assert ctx.measurability and "n_measurable" in ctx.measurability  # #33
    assert isinstance(ctx.proxy_list, list)                        # #34
    assert ctx.missingness is None                                 # no frame supplied


def test_mgb_row_surfaces_proxies():
    ctx = C.build_context(_spec(), "MGB")        # gated -> vitals + mortality are proxies
    concepts = {p["concept"] for p in ctx.proxy_list}
    assert {"map", "death"} <= concepts


def test_corpus_one_row_per_trial_dataset_with_variability():
    results = {"NCT-SEP": [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300),
                           _cr("MGB", 0.95, 0.60, 1.50, 40, 40)]}
    rows = C.build_context_corpus([_spec()], datasets=("MIMIC-IV", "MGB"),
                                  results_by_trial=results)
    assert len(rows) == 2 and {r.dataset for r in rows} == {"MIMIC-IV", "MGB"}
    # #32 variability is trial-level -> attached (denormalized) to every row
    assert all(r.variability and "heterogeneity" in r.variability for r in rows)
    assert all(r.variability["attribution"]["causes"] for r in rows)   # cohort+measurability divergence


def test_missingness_block_when_frame_supplied():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"lactate_max": [1.0, None, 3.0, None]})       # 50% missing
    rows = C.build_context_corpus([_spec()], datasets=("MIMIC-IV",),
                                  frames_by={("NCT-SEP", "MIMIC-IV"): frame})
    assert rows[0].missingness["columns"]["lactate_max"]["missing_fraction"] == 0.5


def test_jsonl_roundtrip_and_join_key(tmp_path):
    rows = C.build_context_corpus([_spec("NCT-A"), _spec("NCT-B")], datasets=("MIMIC-IV", "eICU-CRD"))
    path = tmp_path / "context.jsonl"
    assert dump_context_jsonl(rows, path) == 4                         # 2 trials x 2 datasets
    loaded = list(load_context_jsonl(path))
    assert len(loaded) == 4 and all(isinstance(r, TrialDatasetContext) for r in loaded)
    keys = {(r.nct_id, r.dataset) for r in loaded}
    assert ("NCT-A", "MIMIC-IV") in keys and ("NCT-B", "eICU-CRD") in keys


def test_context_panel_rows_and_rollup():
    from tteEngine.contracts.context import context_panel
    results = {"NCT-SEP": [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300),
                           _cr("MGB", 0.95, 0.60, 1.50, 40, 40)]}
    recs = C.build_context_corpus([_spec()], datasets=("MIMIC-IV", "MGB"),
                                  results_by_trial=results)
    panel = context_panel(recs)
    assert len(panel.rows) == 2
    mimic = next(r for r in panel.rows if r.dataset == "MIMIC-IV")
    assert mimic.emulable and "Emulable" in mimic.why_emulable
    assert mimic.why_divergent and "NCT-SEP" in mimic.why_divergent     # #32 one-liner
    # rollup: MGB not fully measurable -> 1/2 fully; top proxy elements surfaced
    assert panel.rollup.n_rows == 2 and panel.rollup.n_trials == 1
    assert panel.rollup.pct_fully_measurable == 0.5
    assert {e["concept"] for e in panel.rollup.top_proxy_elements} >= {"map", "death"}
    assert panel.rollup.count_by_emulable["emulable"] >= 1


def test_context_panel_joins_comparisons_what_and_why():
    from tteEngine.contracts.context import context_panel
    recs = C.build_context_corpus([_spec()], datasets=("MIMIC-IV",))
    comps = [_cr("MIMIC-IV", 0.62, 0.45, 0.85, 300, 300)]
    row = context_panel(recs, comparisons=comps).rows[0]
    assert row.emulated_estimate == 0.62 and row.agreement is not None   # 'what' joined onto 'why'


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
