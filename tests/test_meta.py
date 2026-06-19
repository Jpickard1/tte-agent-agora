"""Tests for #64 — cross-trial meta-analysis + heterogeneity (probe). Pure:
synthetic ComparisonResults with known answers (no analysis extra / no network)."""

import math

from tteEngine.analysis.meta import (
    concordance_summary,
    meta_analyze,
    pooled_effect,
    random_effects,
    wilson_ci,
)
from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult


def _comp(nct, agreement, est=0.8, lo=0.65, hi=0.98, dataset="MIMIC-IV",
          measure=EffectMeasure.OR) -> ComparisonResult:
    return ComparisonResult(
        nct_id=nct, dataset=dataset, agreement=agreement,
        emulated=TTEResult(nct_id=nct, dataset=dataset, method="iptw", measure=measure,
                           estimate=est, ci_low=lo, ci_high=hi),
    )


def test_wilson_ci():
    lo, hi = wilson_ci(7, 10)
    assert lo < 0.7 < hi and 0.0 <= lo and hi <= 1.0
    assert wilson_ci(0, 0) == (None, None)


def test_concordance_rate_and_ci():
    rows = ([_comp(f"C{i}", Agreement.CONCORDANT) for i in range(7)]
            + [_comp(f"D{i}", Agreement.DISCORDANT) for i in range(3)]
            + [_comp(f"I{i}", Agreement.INCONCLUSIVE) for i in range(2)])
    s = concordance_summary(rows)
    assert s.n == 12 and s.n_comparable == 10 and s.n_concordant == 7
    assert abs(s.rate - 0.7) < 1e-9 and s.ci_low < 0.7 < s.ci_high


def test_random_effects_homogeneous_zero_heterogeneity():
    items = [(math.log(0.8), 0.1)] * 4   # identical -> Q=0, I2=0
    r = random_effects(items, ratio=True)
    assert r.k == 4
    assert abs(r.pooled_estimate - 0.8) < 1e-6
    assert r.i2 == 0.0 and r.tau2 == 0.0


def test_random_effects_detects_heterogeneity():
    items = [(math.log(0.5), 0.08), (math.log(1.0), 0.08), (math.log(2.0), 0.08)]
    r = random_effects(items, ratio=True)
    assert r.i2 > 50.0 and r.tau2 > 0.0    # widely divergent, tight CIs -> high heterogeneity


def test_pooled_effect_skips_unusable():
    rows = [_comp("A", Agreement.CONCORDANT, est=0.8, lo=0.6, hi=1.0),
            _comp("B", Agreement.DISCORDANT, est=float("nan"), lo=None, hi=None)]
    r = pooled_effect(rows)
    assert r.k == 1   # the NaN/no-CI one is dropped


def test_meta_analyze_with_sepsis_subgroup():
    sepsis = {"S1", "S2", "S3"}
    rows = [_comp("S1", Agreement.CONCORDANT), _comp("S2", Agreement.CONCORDANT),
            _comp("S3", Agreement.DISCORDANT),
            _comp("O1", Agreement.CONCORDANT, dataset="eICU"),
            _comp("O2", Agreement.INCONCLUSIVE, dataset="eICU")]
    rep = meta_analyze(rows, subgroup=lambda c: "sepsis" if c.nct_id in sepsis else "other")
    assert rep.overall_concordance.n == 5
    assert len(rep.forest) == 5
    names = {sg.name for sg in rep.by_subgroup}
    assert names == {"sepsis", "other"}
    sep = next(sg for sg in rep.by_subgroup if sg.name == "sepsis")
    assert sep.concordance.n_comparable == 3 and sep.concordance.n_concordant == 2
    assert sep.pooled_effect.k >= 1


def test_corpus_jsonl_roundtrip(tmp_path):
    from tteEngine.analysis.meta import dump_comparisons_jsonl, load_comparisons_jsonl
    rows = [_comp("A", Agreement.CONCORDANT), _comp("B", Agreement.DISCORDANT)]
    p = tmp_path / "corpus.jsonl"
    assert dump_comparisons_jsonl(rows, p) == 2
    loaded = list(load_comparisons_jsonl(p))
    assert [c.nct_id for c in loaded] == ["A", "B"]
    assert loaded[0].agreement == Agreement.CONCORDANT
    # meta_analyze consumes the lazily-loaded stream (offline corpus path)
    rep = meta_analyze(load_comparisons_jsonl(p))
    assert rep.overall_concordance.n == 2
