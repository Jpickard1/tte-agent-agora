"""#63 — the paper skeleton (paper/RESULTS.md) carries the venue-required sections."""
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "paper" / "RESULTS.md"


def test_skeleton_exists_and_has_required_sections():
    assert DOC.exists(), "paper/RESULTS.md missing"
    text = DOC.read_text().lower()
    for section in ("abstract", "method", "results", "limitation", "conclusion"):
        assert section in text, f"missing section: {section}"
    # results tables + flagship + sepsis called out
    for token in ("table 1", "table 2", "table 3", "flagship", "sepsis", "calibration", "concordance"):
        assert token in text, f"missing: {token}"
