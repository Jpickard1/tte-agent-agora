"""Reproducibility: frozen corpus + one-command end-to-end regen (#62, probe).

Makes the whole result reproducible + self-contained. From a VENDORED, frozen
ClinicalTrials.gov snapshot (data/frozen_corpus_studies.jsonl — a pinned,
sepsis-first set with posted results), a single command deterministically
regenerates the corpus of emulated-vs-observed comparisons, the meta-analysis /
calibration / driver outputs, and RESULTS_NARRATIVE.md:

    python -m tteEngine.reproduce            # -> outputs/corpus.jsonl + RESULTS_NARRATIVE.md

Determinism: the OBSERVED side is the trials' real posted results; the EMULATED
side uses a SEEDED synthetic emulator (no EHR data needed, so it reproduces
offline + in CI). Pass `emulate=<real pipeline>` (or run with MIMIC/eICU mounted)
to produce the real numbers — same command, same seeds. Dependency versions are
pinned in pyproject.toml.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from .analysis import (
    compare_trial,
    concordance_drivers,
    corpus_calibration,
    meta_analyze,
    parse_reported_effect,
    write_narrative,
)
from .contracts.io import dump_comparisons_jsonl
from .contracts.results import EffectMeasure, TTEResult
from .ctgov.reader import nct_id_of
from .ctgov.spec import study_to_spec

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CORPUS = REPO_ROOT / "data" / "frozen_corpus_studies.jsonl"
DATASETS = ("MIMIC-IV", "eICU-CRD")
SEED = 0


def load_frozen_studies(path=FROZEN_CORPUS):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _is_sepsis(study: dict) -> bool:
    conds = " ".join(study.get("protocolSection", {})
                     .get("conditionsModule", {}).get("conditions", []) or []).lower()
    return "sepsis" in conds or "septic" in conds


def _rng(nct: str, dataset: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{nct}|{dataset}|{seed}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def synthetic_emulate(study: dict, dataset: str, *, seed: int = SEED) -> TTEResult:
    """Deterministic stand-in emulator for the reproducible OFFLINE regen (no EHR
    data). The emulated OR loosely tracks the trial's reported effect with seeded
    confounding-style bias + a sample-size-driven CI. Replace with the real engine
    (emulate=) when MIMIC/eICU are mounted. Clearly flagged synthetic in .extra."""
    nct = nct_id_of(study) or "NCT?"
    rng = _rng(nct, dataset, seed)
    eff = (parse_reported_effect(study) or {}).get("effect") or {}
    obs = eff.get("risk_ratio")
    log_obs = math.log(obs) if obs and obs > 0 else 0.0
    est = math.exp(log_obs * rng.uniform(0.6, 1.1) + rng.uniform(-0.3, 0.3))
    n = rng.randint(150, 1200)
    se = 0.5 / math.sqrt(n)
    lo, hi = math.exp(math.log(est) - 1.96 * se), math.exp(math.log(est) + 1.96 * se)
    return TTEResult(
        nct_id=nct, dataset=dataset, method="iptw(synthetic)", measure=EffectMeasure.OR,
        estimate=round(est, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
        n_treated=n // 2, n_control=n - n // 2,
        extra={"synthetic": True, "e_value_point": round(1 + abs(math.log(est)), 3)},
    )


def _treatment_hint(study: dict) -> str:
    spec = study_to_spec(study)
    treated = next((a for a in spec.arms if not a.is_control), None)
    return " ".join(treated.intervention_concepts) if treated else ""


def reproduce(*, corpus_path=FROZEN_CORPUS, out_dir="outputs", seed: int = SEED,
              datasets=DATASETS, emulate=None) -> dict:
    """Deterministically regenerate corpus -> JSONL -> meta/calibration/drivers ->
    RESULTS_NARRATIVE.md from the frozen ctgov snapshot. `emulate(study, dataset)
    -> TTEResult` defaults to the seeded synthetic emulator (offline-reproducible)."""
    studies = list(load_frozen_studies(corpus_path))
    sepsis_ncts = {nct_id_of(s) for s in studies if _is_sepsis(s)}
    used_synthetic = emulate is None
    emulate = emulate or (lambda st, ds: synthetic_emulate(st, ds, seed=seed))

    comparisons = []
    for s in studies:
        hint = _treatment_hint(s)
        for ds in datasets:
            comparisons.append(compare_trial(s, emulate(s, ds), treatment_hint=hint, dataset=ds))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_comparisons_jsonl(comparisons, out / "corpus.jsonl")
    meta = meta_analyze(comparisons,
                        subgroup=lambda c: "sepsis" if c.nct_id in sepsis_ncts else "other")
    cal = corpus_calibration(comparisons)
    drivers = concordance_drivers(comparisons, sepsis_fn=lambda c: c.nct_id in sepsis_ncts)
    (out / "RESULTS_NARRATIVE.md").write_text(write_narrative(drivers, meta=meta, calibration=cal))
    return {
        "n_studies": len(studies),
        "n_comparisons": len(comparisons),
        "concordance_rate": meta.overall_concordance.rate,
        "calibration_slope": cal.slope,
        "i2": meta.pooled_effect.i2,
        "out_dir": str(out),
        "synthetic_emulation": used_synthetic,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reproducible end-to-end TTE corpus regen (#62)")
    ap.add_argument("--corpus", default=str(FROZEN_CORPUS))
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    print(json.dumps(reproduce(corpus_path=a.corpus, out_dir=a.out, seed=a.seed), indent=2))


if __name__ == "__main__":
    main()
