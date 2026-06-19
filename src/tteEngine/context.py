"""Context builder (#95) — assemble the per-(nct_id, dataset) WHY record from the
rigor layers and persist it as a sidecar to the corpus JSONL.

Bundles, per (trial, dataset):
  * #35 triage    -> emulability score + sepsis flag + reasons,
  * #33 measurability -> per-element measurable/proxy/unmeasurable summary,
  * #34 missingness   -> proxy list (+ data missingness when a frame is given),
  * #32 variability   -> the TRIAL-level cross-dataset heterogeneity + attribution
                         (why concordant/divergent), denormalized onto each row.

Additive: this does NOT touch the corpus runner (corpus.py) — call
`write_context_sidecar(specs, path, ...)` after a corpus run to drop a
`context.jsonl` next to the corpus, joined by (nct_id, dataset).

Import-light at module load (triage/measurability/missingness/variability are all
import-light; pandas only if a frame is passed, #64 meta only if variability is
computed) -> runs in CI's [dev] env.
"""
from __future__ import annotations

from . import measurability, missingness, triage, variability
from .contracts.context import TrialDatasetContext, dump_context_jsonl
from .contracts.trial_spec import TargetTrialSpec

DEFAULT_DATASETS = ("MIMIC-IV", "eICU-CRD")


def build_context(
    spec: TargetTrialSpec, dataset: str, *,
    frame=None, feature_columns=None, threshold: float = 0.5,
    variability_block: dict | None = None,
) -> TrialDatasetContext:
    """The WHY record for one (trial, dataset). `frame` (optional analysis frame)
    adds the data-missingness block; `variability_block` (optional, trial-level)
    is the #32 report shared across the trial's dataset rows."""
    score = triage.score_spec(spec, dataset, threshold=threshold)
    meas = measurability.measurability_report(spec, dataset).summary
    proxies = missingness.proxy_substitution_list(spec, dataset)
    miss = missingness.missingness_summary(frame, feature_columns) if frame is not None else None
    return TrialDatasetContext(
        nct_id=spec.nct_id, dataset=dataset, is_sepsis=score.is_sepsis,
        emulable=score.emulable, emulability_score=score.score,
        emulability=score.model_dump(), measurability=meas,
        proxy_list=proxies, missingness=miss, variability=variability_block,
    )


def build_context_corpus(
    specs, *, datasets: tuple[str, ...] = DEFAULT_DATASETS,
    results_by_trial: dict | None = None, frames_by: dict | None = None,
    threshold: float = 0.5,
) -> list[TrialDatasetContext]:
    """One TrialDatasetContext per (trial, dataset).

    `results_by_trial` ({nct_id: [ComparisonResult per dataset]}) drives the #32
    variability block per trial. `frames_by` ({(nct_id, dataset): analysis_frame})
    drives the #34 missingness block per row. Both optional — without them the
    record still carries emulability + measurability + proxy list (spec-only)."""
    out: list[TrialDatasetContext] = []
    for spec in specs:
        var_block = None
        results = (results_by_trial or {}).get(spec.nct_id)
        if results:
            var_block = variability.variability_report(spec, results)
        for ds in datasets:
            frame = (frames_by or {}).get((spec.nct_id, ds))
            out.append(build_context(spec, ds, frame=frame, threshold=threshold,
                                     variability_block=var_block))
    return out


def write_context_sidecar(specs, path, **kwargs) -> int:
    """Build the context corpus and persist it as a JSONL sidecar next to the
    corpus (one record per (trial, dataset)). Returns the count written. Call this
    after a corpus run; it does not touch corpus.py."""
    return dump_context_jsonl(build_context_corpus(specs, **kwargs), path)


__all__ = ["build_context", "build_context_corpus", "write_context_sidecar", "DEFAULT_DATASETS"]
