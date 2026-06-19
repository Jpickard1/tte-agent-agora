"""#105 confounder adjustability ledger + PS-overlap surfacing.

Pure (measurability is import-light); no analysis extra needed for the ledger
logic. A separate case checks the engine surfaces ps_overlap + covariates_used
into TTEResult.extra (needs pandas/analysis -> importorskip).
"""
import sys
from pathlib import Path

import pytest

from tteEngine.adjustability import (
    STANDARD_ICU_CONFOUNDERS,
    build_ledger_corpus,
    confounder_ledger,
    write_ledger_sidecar,
)
from tteEngine.contracts.adjustability import (
    Adjustability,
    ConfounderLedger,
    load_ledger_jsonl,
)
from tteEngine.contracts.results import (
    Agreement,
    ComparisonResult,
    EffectMeasure,
    TTEResult,
)
from tteEngine.contracts.trial_spec import TargetTrialSpec


def _spec(nct="NCT1"):
    return TargetTrialSpec(nct_id=nct, title="t")


def test_classification_adjusted_measurable_unused_notadjustable():
    # With the default (#33) verdicts in MIMIC-IV: labs are measurable, demographics
    # are proxy-only (adapters don't emit them yet). So a LAB in the model -> ADJUSTED;
    # a measurable one NOT in the model -> MEASURABLE_NOT_USED; a proxy one (age) ->
    # NOT_ADJUSTABLE even though a column was thrown in (proxy = residual confounding).
    # Covariate columns carry feature suffixes (e.g. lactate_max) -> substring match.
    led = confounder_ledger(
        _spec(), "MIMIC-IV",
        covariates_used=["lactate_max", "creatinine", "age"],
        balance_rows=[{"variable": "lactate_max", "smd_before": 0.4, "smd_after": 0.05}],
        overlap={"poor": False, "overlap_coef": 0.8, "frac_treated_off_support": 0.01},
        e_value_point=1.6, adjustment="iptw",
    )
    assert isinstance(led, ConfounderLedger)
    assert led.n_considered == len(STANDARD_ICU_CONFOUNDERS)
    by = {r.confounder: r for r in led.considered}
    assert by["lactate"].classification == Adjustability.ADJUSTED and by["lactate"].in_model
    assert by["lactate"].smd_after == 0.05            # SMD matched via the _max-suffix column
    assert by["creatinine"].classification == Adjustability.ADJUSTED
    assert by["bilirubin"].classification == Adjustability.MEASURABLE_NOT_USED  # measurable, not in model
    assert by["age"].classification == Adjustability.NOT_ADJUSTABLE             # proxy-only here
    # the three buckets partition the considered set
    assert led.n_adjusted + led.n_measurable_not_used + led.n_not_adjustable == led.n_considered
    assert led.n_adjusted >= 2
    assert "adjusted" in led.summary_line


def test_not_adjustable_ties_to_evalue_in_note():
    led = confounder_ledger(
        _spec(), "MIMIC-IV", covariates_used=[], e_value_point=1.45, adjustment="iptw")
    # nothing in the model -> measurable ones unused, unmeasurable/proxy -> not_adjustable
    assert led.n_adjusted == 0
    if led.n_not_adjustable:
        assert "residual confounding" in led.residual_confounding_note
        assert "1.45" in led.residual_confounding_note


def test_measure_fn_injectable_overrides_default():
    # force everything unmeasurable -> all NOT_ADJUSTABLE
    led = confounder_ledger(
        _spec(), "MIMIC-IV", covariates_used=["age"],
        measure_fn=lambda c, et, ds: ("unmeasurable", "forced"))
    assert led.n_not_adjustable == led.n_considered
    assert all(r.classification == Adjustability.NOT_ADJUSTABLE for r in led.considered)


def _comp(nct, covs):
    return ComparisonResult(
        nct_id=nct, dataset="eICU-CRD", agreement=Agreement.CONCORDANT,
        observed_estimate=0.9, observed_measure=EffectMeasure.RR,
        emulated=TTEResult(nct_id=nct, dataset="eICU-CRD", method="IPTW",
                           measure=EffectMeasure.OR, estimate=0.7,
                           extra={"covariates_used": covs, "e_value_point": 1.5,
                                  "ps_overlap": {"poor": True, "overlap_coef": 0.4,
                                                 "frac_treated_off_support": 0.2},
                                  "balance": [{"variable": "age", "smd_before": 0.3,
                                               "smd_after": 0.04}]}))


def test_sidecar_roundtrip_joins_on_key(tmp_path):
    specs = [_spec("NCT1"), _spec("NCT2")]
    comps = [_comp("NCT1", ["age", "creatinine"]), _comp("NCT2", ["lactate"])]
    n = write_ledger_sidecar(comps, specs, tmp_path / "ledger.jsonl")
    assert n == 2
    back = list(load_ledger_jsonl(tmp_path / "ledger.jsonl"))
    assert {(l.nct_id, l.dataset) for l in back} == {("NCT1", "eICU-CRD"), ("NCT2", "eICU-CRD")}
    # ps_overlap rides through; poor overlap noted in the summary
    led = next(l for l in back if l.nct_id == "NCT1")
    assert led.ps_overlap["poor"] is True
    assert "PS overlap" in led.summary_line


def test_engine_surfaces_ps_overlap_and_covariates_used(tmp_path):
    pytest.importorskip("pandas")
    pytest.importorskip("lifelines")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    import sepsis_vignette as vig

    from tteEngine.cohort import build_cohort
    from tteEngine.orchestration.engine_provider import make_engine_provider

    events = vig.confounded_stream(scale=2)
    spec = vig.demo_spec()
    cohort = build_cohort(events, spec, dataset="MIMIC-IV")
    provider = make_engine_provider([vig.LACTATE], adjustment="iptw")
    res = provider(events, cohort, spec)
    assert any("lactate" in c.lower() for c in res.extra.get("covariates_used", []))
    ov = res.extra.get("ps_overlap")
    assert ov is not None and "overlap_coef" in ov and "bin_centers" in ov
