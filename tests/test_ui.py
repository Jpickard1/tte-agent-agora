"""#49 results dashboard + #40 Trial Emulation Cards — the pure data layer that
the Streamlit app renders. Needs the analysis extra for build_dashboard."""

import pytest

from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult
from tteEngine.ui import build_cards


def _cr(nct, dataset, est, agreement, observed=0.9, measure=EffectMeasure.OR):
    return ComparisonResult(
        nct_id=nct, dataset=dataset,
        emulated=TTEResult(nct_id=nct, dataset=dataset, method="iptw", measure=measure,
                           estimate=est, ci_low=est * 0.8, ci_high=est * 1.2,
                           n_treated=20, n_control=20, extra={"p_value": 0.03, "e_value_point": 1.9}),
        observed_estimate=observed, observed_measure=EffectMeasure.RR, agreement=agreement,
    )


# ---- #40 cards (pure, import-light) ----

def test_build_cards_verdict_and_sepsis_flag():
    cards = build_cards(
        [_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT),
         _cr("NCT2", "eICU-CRD", 1.4, Agreement.DISCORDANT)],
        sepsis_ncts={"NCT1"})
    c1, c2 = cards
    assert c1.is_sepsis is True and c2.is_sepsis is False
    assert "AGREES" in c1.verdict and "benefit" in c1.verdict   # OR<1 -> benefit
    assert "DISAGREES" in c2.verdict
    assert c1.p_value == 0.03 and c1.e_value == 1.9
    assert c1.n_treated == 20


def test_cards_are_import_light():
    # fresh interpreter: importing + using the card layer must NOT pull matplotlib
    # or the analysis engine (so the UI renders without those deps).
    import subprocess
    import sys
    code = (
        "import sys; from tteEngine.ui.cards import build_cards; "
        "assert 'matplotlib' not in sys.modules, 'cards pulled matplotlib'; "
        "assert 'lifelines' not in sys.modules, 'cards pulled lifelines'; "
        "print('import-light OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"})
    assert "import-light OK" in r.stdout, r.stderr


# ---- #49 dashboard (needs analysis) ----

def test_build_dashboard_assembles_headline_forest_calibration_cards():
    pytest.importorskip("numpy")
    pytest.importorskip("statsmodels")
    from tteEngine.ui import build_dashboard

    rows = ([_cr(f"S{i}", "MIMIC-IV", 0.6, Agreement.CONCORDANT) for i in range(5)] +
            [_cr(f"O{i}", "eICU-CRD", 1.5, Agreement.DISCORDANT) for i in range(3)])
    m = build_dashboard(rows, sepsis_ncts={f"S{i}" for i in range(5)})
    assert m.n_total == 8 and m.n_sepsis == 5
    # concordance: 5 concordant of 8 comparable
    assert m.concordance["n_comparable"] == 8 and m.concordance["n_concordant"] == 5
    assert m.concordance["rate"] == pytest.approx(5 / 8)
    assert "i2" in m.pooled
    assert "slope" in m.calibration and isinstance(m.calibration["points"], list)
    assert len(m.forest_rows) == 8 and len(m.cards) == 8
    # sepsis subgroup pooled present
    assert m.sepsis_pooled is not None and m.sepsis_pooled["k"] == 5


# ---- thin Streamlit app chart builders (web extra) ----

def test_results_app_chart_builders():
    pytest.importorskip("altair")
    pytest.importorskip("pandas")
    pytest.importorskip("streamlit")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
    import results_app as app
    from tteEngine.figures import forest_rows

    rows = forest_rows([_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT),
                        _cr("NCT2", "eICU-CRD", 1.4, Agreement.DISCORDANT)])
    assert app._forest_chart(rows) is not None
    assert app._forest_chart([]) is None
    assert app._calibration_chart([{"emulated": 0.6, "observed": 0.7, "in_ci": True}]) is not None
    assert app._calibration_chart([]) is None


# ---- #98 WHY-context panel (joins worker1's #95 sidecar) ----

def _ctx(nct, dataset, *, emulable=True, score=0.8, is_sepsis=False, n_meas=8, n_proxy=2,
         n_unmeas=1, gaps=None, proxy=None, variability=None):
    from tteEngine.contracts.context import TrialDatasetContext
    return TrialDatasetContext(
        nct_id=nct, dataset=dataset, is_sepsis=is_sepsis, emulable=emulable,
        emulability_score=score,
        measurability={"n_measurable": n_meas, "n_proxy": n_proxy, "n_unmeasurable": n_unmeas,
                       "fully_measurable": n_unmeas == 0, "gaps": gaps or []},
        proxy_list=[{"element": e} for e in (proxy or [])],
        variability=variability,
    )


def test_why_for_emulable_and_proxy():
    from tteEngine.ui import why_for
    w = why_for(_ctx("NCT1", "MIMIC-IV", proxy=["sofa", "map"]))
    assert "Emulable" in w["why_emulable"]
    assert w["measurability"]["n_proxy"] == 2
    assert w["proxy_elements"] == ["sofa", "map"]


def test_why_for_not_emulable_lists_gaps():
    from tteEngine.ui import why_for
    w = why_for(_ctx("NCT2", "eICU-CRD", emulable=False, score=0.2, n_unmeas=3,
                     gaps=["primary_outcome", "key_inclusion"]))
    assert "Not emulable" in w["why_emulable"] and "primary_outcome" in w["why_emulable"]


def test_why_for_divergence_from_variability():
    from tteEngine.ui import why_for
    w = why_for(_ctx("NCT3", "MIMIC-IV",
                     variability={"heterogeneity": {"i2": 0.62},
                                  "attribution": {"causes": ["case-mix"], "note": "eICU sicker"}}))
    assert "I²=62%" in w["why_divergent"] and "case-mix" in w["why_divergent"]


def test_corpus_context_summary_rollup():
    from tteEngine.ui import corpus_context_summary
    recs = [_ctx("S1", "MIMIC-IV", is_sepsis=True, n_unmeas=0, proxy=["sofa"]),
            _ctx("O1", "eICU-CRD", emulable=False, score=0.1, proxy=["sofa", "map"])]
    s = corpus_context_summary(recs)
    assert s["n"] == 2 and s["n_emulable"] == 1 and s["n_sepsis"] == 1
    assert s["pct_fully_measurable"] == 0.5
    assert s["top_proxy_elements"][0] == {"element": "sofa", "n": 2}


def test_build_dashboard_joins_context_onto_cards():
    pytest.importorskip("numpy")
    pytest.importorskip("statsmodels")
    from tteEngine.ui import build_dashboard
    rows = [_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT),
            _cr("NCT2", "eICU-CRD", 1.4, Agreement.DISCORDANT)]
    ctx = [_ctx("NCT1", "MIMIC-IV", proxy=["sofa"]), _ctx("NCT2", "eICU-CRD", emulable=False)]
    m = build_dashboard(rows, context_records=ctx)
    by = {(c.nct_id, c.dataset): c for c in m.cards}
    assert by[("NCT1", "MIMIC-IV")].why["emulable"] is True
    assert by[("NCT2", "eICU-CRD")].why["emulable"] is False
    assert m.context_summary["n"] == 2
