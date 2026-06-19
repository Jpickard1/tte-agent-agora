"""#105 support: per-confounder measurability verdict (the injectable measure_fn
probe's adjustability ledger joins on). Pure stdlib; frame path guards pandas."""

import pytest

from tteEngine import measurability as M
from tteEngine.contracts.events import EventType


def test_lab_confounder_measurable_in_icu_dbs():
    for ds in ("MIMIC-IV", "eICU-CRD"):
        v = M.confounder_measurability("lactate", EventType.LAB, ds)
        assert v["status"] == M.MEASURABLE and v["confounder"] == "lactate" and v["dataset"] == ds


def test_vitals_confounder_proxy_in_mgb():
    v = M.confounder_measurability("map", EventType.MEASUREMENT, "MGB")   # gated -> proxy
    assert v["status"] == M.PROXY and "needs wiring" in v["reason"]


def test_soft_confounder_unmeasurable():
    v = M.confounder_measurability("quality of life", EventType.OUTCOME, "MIMIC-IV")
    assert v["status"] == M.UNMEASURABLE


def test_missing_fraction_when_frame_supplied():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"lactate_max": [1.0, None, None, 4.0]})        # 50% missing
    v = M.confounder_measurability("lactate", EventType.LAB, "MIMIC-IV",
                                   frame=frame, column="lactate_max")
    assert v["status"] == M.MEASURABLE and v["missing_fraction"] == 0.5


def test_usable_as_injectable_measure_fn():
    # probe's pattern: measure_fn(concept, event_type, dataset) -> verdict
    considered = [("age", EventType.DEMOGRAPHIC), ("lactate", EventType.LAB),
                  ("map", EventType.MEASUREMENT)]
    verdicts = [M.confounder_measurability(c, et, "eICU-CRD") for c, et in considered]
    assert {v["status"] for v in verdicts} <= {M.MEASURABLE, M.PROXY, M.UNMEASURABLE}
    assert all("confounder" in v and "reason" in v for v in verdicts)


def run():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, t in tests:
        t(); print("PASS", name)
    print(f"\n{len(tests)}/{len(tests)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
