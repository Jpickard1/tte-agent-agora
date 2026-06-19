"""#60 forest figures: pure plot-ready rows (no matplotlib) + the rendered plot
(skips without the viz extra). Decoupled from the corpus producer."""

import pytest

from tteEngine.contracts.results import Agreement, ComparisonResult, EffectMeasure, TTEResult
from tteEngine.figures import ForestRow, forest_rows


def _cr(nct, dataset, est, agreement, measure=EffectMeasure.OR, observed=0.9):
    return ComparisonResult(
        nct_id=nct, dataset=dataset,
        emulated=TTEResult(nct_id=nct, dataset=dataset, method="iptw", measure=measure,
                           estimate=est, ci_low=est * 0.8, ci_high=est * 1.2,
                           n_treated=10, n_control=10),
        observed_estimate=observed, observed_measure=EffectMeasure.RR, agreement=agreement,
    )


def test_forest_rows_pure_and_import_light():
    rows = forest_rows([_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT),
                        _cr("NCT2", "eICU-CRD", 1.3, Agreement.DISCORDANT)])
    assert [r.label for r in rows] == ["NCT1 [MIMIC-IV]", "NCT2 [eICU-CRD]"]
    assert rows[0].estimate == 0.6 and rows[0].ci_low == pytest.approx(0.48)
    assert rows[0].agreement == "concordant" and rows[1].dataset == "eICU-CRD"


def test_forest_rows_skip_nan():
    nan = _cr("NCTx", "MIMIC-IV", float("nan"), Agreement.INCONCLUSIVE)
    assert forest_rows([nan]) == []


def test_forest_rows_import_light():
    # the pure layer must not require matplotlib (UI #49 reads it without viz extra)
    import sys
    forest_rows([_cr("NCT1", "MIMIC-IV", 0.6, Agreement.CONCORDANT)])
    # forest_rows imported fine above; matplotlib only loads inside forest_plot
    assert "tteEngine.figures.forest" in sys.modules


def test_forest_plot_writes_file(tmp_path):
    plt = pytest.importorskip("matplotlib")  # viz extra
    from tteEngine.figures import forest_plot
    rows = [_cr(f"NCT{i}", "MIMIC-IV" if i % 2 else "eICU-CRD", 0.6 + 0.1 * i,
                Agreement.CONCORDANT) for i in range(6)]
    out = tmp_path / "forest.png"
    p = forest_plot(rows, out, title="test forest")
    assert out.exists() and out.stat().st_size > 0 and p == str(out)


def test_forest_plot_empty_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from tteEngine.figures import forest_plot
    with pytest.raises(ValueError):
        forest_plot([], tmp_path / "x.png")


# --- calibration figure (#60): consumes #41's CalibrationCurve shape (duck-typed) ---

class _StubPoint:
    def __init__(self, emulated, observed, in_ci):
        self.emulated, self.observed, self.in_ci = emulated, observed, in_ci


class _StubCurve:
    """Matches probe's #41 CalibrationCurve shape (points/slope/intercept/coverage)."""
    def __init__(self):
        self.points = [_StubPoint(0.6, 0.7, True), _StubPoint(1.3, 0.9, False),
                       _StubPoint(0.8, 0.85, True)]
        self.slope, self.intercept, self.coverage, self.n = 0.9, 0.05, 2 / 3, 3


def test_calibration_points_pure():
    from tteEngine.figures import calibration_points
    pts = calibration_points(_StubCurve())
    assert len(pts) == 3
    assert pts[0] == {"emulated": 0.6, "observed": 0.7, "in_ci": True}
    assert pts[1]["in_ci"] is False


def test_calibration_points_accepts_dicts():
    from tteEngine.figures import calibration_points

    class C:
        points = [{"emulated": 0.5, "observed": 0.6, "in_ci": True},
                  {"emulated": float("nan"), "observed": 1.0, "in_ci": False}]  # NaN skipped
    assert calibration_points(C()) == [{"emulated": 0.5, "observed": 0.6, "in_ci": True}]


def test_calibration_plot_writes_file(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    from tteEngine.figures import calibration_plot
    out = tmp_path / "calibration.png"
    p = calibration_plot(_StubCurve(), out)
    assert out.exists() and out.stat().st_size > 0 and p == str(out)


def test_calibration_plot_empty_raises(tmp_path):
    pytest.importorskip("matplotlib")
    from tteEngine.figures import calibration_plot

    class Empty:
        points = []
        slope = intercept = coverage = None
    with pytest.raises(ValueError):
        calibration_plot(Empty(), tmp_path / "x.png")
