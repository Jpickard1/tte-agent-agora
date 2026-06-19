"""#62 — reproducible regen: frozen corpus + deterministic one-command pipeline.
Runs OFFLINE (synthetic emulator, no EHR data / no analysis extra)."""
from pathlib import Path

from tteEngine.contracts.context import load_context_jsonl
from tteEngine.contracts.io import load_comparisons_jsonl
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


def test_reproduce_emits_context_sidecar_joined_to_corpus(tmp_path):
    """#95 WHY sidecar: one record per (nct_id, dataset), same join key as corpus."""
    s = reproduce(out_dir=tmp_path, seed=0)
    assert (tmp_path / "context.jsonl").exists()
    ctx = list(load_context_jsonl(tmp_path / "context.jsonl"))
    corp = list(load_comparisons_jsonl(tmp_path / "corpus.jsonl"))
    assert s["n_context"] == len(ctx) == len(corp)
    assert {(r.nct_id, r.dataset) for r in ctx} == {(c.nct_id, c.dataset) for c in corp}
    assert all(r.emulability_score is not None for r in ctx)


def test_context_sidecar_deterministic_and_optional(tmp_path):
    reproduce(out_dir=tmp_path / "a", seed=0)
    reproduce(out_dir=tmp_path / "b", seed=0)
    assert (tmp_path / "a" / "context.jsonl").read_text() == (tmp_path / "b" / "context.jsonl").read_text()
    s = reproduce(out_dir=tmp_path / "c", seed=0, context=False)
    assert s["n_context"] == 0 and not (tmp_path / "c" / "context.jsonl").exists()
