"""#62 — reproducible regen: frozen corpus + deterministic one-command pipeline.
Runs OFFLINE (synthetic emulator, no EHR data / no analysis extra)."""
from pathlib import Path

from tteEngine.ctgov.reader import nct_id_of
from tteEngine.reproduce import load_frozen_studies, reproduce, synthetic_emulate


def test_frozen_corpus_vendored():
    studies = list(load_frozen_studies())
    assert len(studies) >= 5 and all(nct_id_of(s) for s in studies)


def test_synthetic_emulate_deterministic():
    s = list(load_frozen_studies())[0]
    a, b = synthetic_emulate(s, "MIMIC-IV", seed=0), synthetic_emulate(s, "MIMIC-IV", seed=0)
    assert (a.estimate, a.ci_low, a.ci_high, a.n_treated) == (b.estimate, b.ci_low, b.ci_high, b.n_treated)
    # different dataset/seed -> different draw
    assert synthetic_emulate(s, "eICU-CRD", seed=0).estimate != a.estimate


def test_reproduce_is_deterministic(tmp_path):
    s1 = reproduce(out_dir=tmp_path / "r1", seed=0)
    s2 = reproduce(out_dir=tmp_path / "r2", seed=0)
    assert s1["n_comparisons"] == s2["n_comparisons"] == s1["n_studies"] * 2
    assert (tmp_path / "r1" / "RESULTS_NARRATIVE.md").read_text() == (tmp_path / "r2" / "RESULTS_NARRATIVE.md").read_text()
    assert (tmp_path / "r1" / "corpus.jsonl").read_text() == (tmp_path / "r2" / "corpus.jsonl").read_text()


def test_reproduce_outputs_and_summary(tmp_path):
    s = reproduce(out_dir=tmp_path, seed=0)
    assert (tmp_path / "corpus.jsonl").exists() and (tmp_path / "RESULTS_NARRATIVE.md").exists()
    assert s["concordance_rate"] is not None
    assert "## Headline" in (tmp_path / "RESULTS_NARRATIVE.md").read_text()
