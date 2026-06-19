"""Tests for #61 — concordance-driver analysis + narrative (probe). Pure:
synthetic ComparisonResults where a feature clearly drives concordance."""

from tteEngine.analysis import (
    concordance_drivers,
    corpus_calibration,
    meta_analyze,
    write_narrative,
)
from tteEngine.analysis.drivers import DriverReport
from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult


def _comp(nct, agreement, dataset="MIMIC-IV", est=0.8, measure=EffectMeasure.OR, n=400):
    return ComparisonResult(
        nct_id=nct, dataset=dataset, agreement=agreement,
        observed_estimate=0.8, observed_measure=EffectMeasure.RR,
        emulated=TTEResult(nct_id=nct, dataset=dataset, method="iptw", measure=measure,
                           estimate=est, ci_low=est * 0.7, ci_high=est * 1.4,
                           n_treated=n // 2, n_control=n // 2, extra={"e_value_point": 1.5}),
    )


def test_dataset_is_top_driver():
    rows = ([_comp(f"M{i}", Agreement.CONCORDANT, "MIMIC-IV") for i in range(5)]
            + [_comp(f"E{i}", Agreement.DISCORDANT, "eICU") for i in range(5)])
    rep = concordance_drivers(rows)
    assert isinstance(rep, DriverReport)
    assert rep.n_comparable == 10 and abs(rep.overall_concordance - 0.5) < 1e-9
    top = rep.associations[0]
    assert top.feature == "dataset" and abs(top.spread - 1.0) < 1e-9
    rates = {s["level"]: s["concordance_rate"] for s in top.strata}
    assert rates["MIMIC-IV"] == 1.0 and rates["eICU"] == 0.0


def test_continuous_effect_magnitude_driver():
    # large effect -> concordant, near-null -> discordant (only effect_magnitude varies)
    rows = ([_comp(f"L{i}", Agreement.CONCORDANT, est=0.4) for i in range(5)]
            + [_comp(f"S{i}", Agreement.DISCORDANT, est=0.95) for i in range(5)])
    rep = concordance_drivers(rows)
    em = next(a for a in rep.associations if a.feature == "effect_magnitude")
    assert em.kind == "continuous" and em.spread is not None and em.spread > 0.5


def test_sepsis_finding():
    rows = ([_comp(f"S{i}", Agreement.CONCORDANT) for i in range(4)]
            + [_comp(f"O{i}", Agreement.DISCORDANT) for i in range(4)])
    sepsis = {f"S{i}" for i in range(4)}
    rep = concordance_drivers(rows, sepsis_fn=lambda c: c.nct_id in sepsis)
    assert rep.sepsis_finding and "100%" in rep.sepsis_finding


def test_write_narrative():
    rows = ([_comp(f"M{i}", Agreement.CONCORDANT) for i in range(7)]
            + [_comp(f"E{i}", Agreement.DISCORDANT, "eICU") for i in range(3)])
    md = write_narrative(concordance_drivers(rows), meta=meta_analyze(rows),
                         calibration=corpus_calibration(rows))
    assert "# TTE Emulation" in md and "## Headline" in md
    assert "concordance" in md.lower() and "drivers" in md.lower()
