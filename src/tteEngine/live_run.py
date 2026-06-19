"""Live-run driver (#102, probe): real MIMIC-IV + eICU corpus -> the real numbers.

Composes the already-merged, tested spine into ONE command for the live run on
exxact07 (where MIMIC/eICU are local):

    fetch_corpus(sepsis_first)             # #1/#58 ctgov catalog (explicit max_studies cap)
    -> study_to_spec                       # #2
    -> keep trials emulable in >=1 dataset # #35 score_spec (kept-but-LOGGED, never silent)
    -> run_corpus(extract_fn=<#101 loader>, engine_fn=make_engine_provider(IPTW))
                                           # #36 spine: extract -> cohort -> REAL engine,
                                           #     streaming, with a DropLog (no silent caps)
    -> corpus.jsonl + context.jsonl        # #84 persist + #95 WHY sidecar
    -> meta / calibration / drivers        # #64 / #41 / #61
    -> RESULTS_NARRATIVE.md (+ forest fig) # #61 / #60

`extract_fn` is INJECTED — that is the one live seam:
  * REAL run: worker1's #101 MIMIC/eICU loader, `extract_fn(plan, spec, dataset)
    -> canonical 5-col DataFrame | None`.
  * SCAFFOLD (until #101 lands): `--synthetic` routes the #13 vignette's
    confounded 5-col stream through the REAL cohort builder + REAL IPTW engine, so
    the whole driver is validated end-to-end (incl. the confounding flip) offline.

NO SILENT CAPS: `max_studies` is the only cap and it is reported in the summary;
every (trial, dataset) that can't emulate is recorded in a DropLog (drops.jsonl)
with a reason; trials that aren't emulable are counted, not hidden.

This module only ORCHESTRATES; cohort/engine/persistence/analysis/figures are the
existing tested machinery. Heavy deps (analysis extra) load lazily via those.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .context import write_context_sidecar
from .contracts.io import load_comparisons_jsonl
from .orchestration.corpus import DropLog, run_corpus_to_jsonl
from .orchestration.engine_provider import make_engine_provider
from .triage import score_spec

DATASETS = ("MIMIC-IV", "eICU-CRD")


def build_emulable_jobs(*, max_studies, datasets=DATASETS, sepsis_first=True,
                        threshold=0.5, emulable_only=True, http_get=None):
    """Fetch the sepsis-first ctgov catalog and keep trials emulable in >=1 dataset.

    Returns (jobs, specs, catalog). jobs = [(study_dict, TargetTrialSpec)] for
    run_corpus. `catalog` reports n_fetched / n_emulable / n_unemulable / n_sepsis
    and the explicit max_studies cap — coverage is never silently overstated."""
    from .ctgov import fetch_corpus, study_to_spec

    kw = {"http_get": http_get} if http_get is not None else {}
    studies = fetch_corpus(max_studies=max_studies, sepsis_first=sepsis_first, **kw)
    jobs: list[tuple[dict, object]] = []
    specs: list = []
    n_unparseable = n_unemulable = n_sepsis = 0
    for s in studies:
        try:
            spec = study_to_spec(s)
        except Exception:
            n_unparseable += 1
            continue
        scores = [score_spec(spec, ds, threshold=threshold) for ds in datasets]
        if emulable_only and not any(sc.emulable for sc in scores):
            n_unemulable += 1
            continue
        jobs.append((s, spec))
        specs.append(spec)
        if scores and scores[0].is_sepsis:
            n_sepsis += 1
    catalog = {
        "n_fetched": len(studies),
        "n_unparseable": n_unparseable,
        "n_unemulable": n_unemulable,
        "n_emulable": len(jobs),
        "n_sepsis_emulable": n_sepsis,
        "max_studies": max_studies,
        "datasets": list(datasets),
        "sepsis_first": sepsis_first,
        "threshold": threshold,
    }
    return jobs, specs, catalog


def _write_analysis(comparisons, specs, out, *, datasets, context=True, figures=False):
    """corpus -> #64 meta / #41 calibration / #61 drivers -> RESULTS_NARRATIVE.md,
    + the #95 context.jsonl sidecar (joined on (nct_id,dataset)) and the #60 forest
    figure. Same building blocks as reproduce.py, so the live gallery == the frozen
    gallery in shape."""
    from .analysis import (
        concordance_drivers,
        corpus_calibration,
        meta_analyze,
        write_narrative,
    )

    sepsis_ncts = {sp.nct_id for sp in specs if score_spec(sp, datasets[0]).is_sepsis}
    meta = meta_analyze(comparisons,
                        subgroup=lambda c: "sepsis" if c.nct_id in sepsis_ncts else "other")
    cal = corpus_calibration(comparisons)
    drivers = concordance_drivers(comparisons, sepsis_fn=lambda c: c.nct_id in sepsis_ncts)
    (out / "RESULTS_NARRATIVE.md").write_text(write_narrative(drivers, meta=meta, calibration=cal))

    n_context = 0
    if context:
        results_by_trial: dict = {}
        for c in comparisons:
            results_by_trial.setdefault(c.nct_id, []).append(c)
        n_context = write_context_sidecar(specs, out / "context.jsonl",
                                          datasets=tuple(datasets), results_by_trial=results_by_trial)

    figure_path = None
    if figures:
        try:
            from .figures.forest import forest_plot
            figure_path = forest_plot(comparisons, str(out / "forest.png"))
        except Exception as exc:  # viz extra absent / empty corpus -> never break the run
            figure_path = f"skipped: {type(exc).__name__}: {exc}"

    return {
        "concordance_rate": meta.overall_concordance.rate,
        "calibration_slope": cal.slope,
        "i2": meta.pooled_effect.i2,
        "n_context": n_context,
        "forest_figure": figure_path,
    }


def run_live(*, extract_fn, engine_fn=None, compare_fn=None, jobs=None, specs=None,
             out_dir="live_outputs", datasets=DATASETS, max_studies=2000,
             adjustment="iptw", covariates=None, threshold=0.5, emulable_only=True,
             http_get=None, context=True, figures=True) -> dict:
    """Run the live corpus end-to-end and write the gallery artifacts to out_dir:
    corpus.jsonl, context.jsonl, RESULTS_NARRATIVE.md, drops.jsonl, summary.json
    (+ forest.png). `jobs`/`specs` may be injected (tests / a pre-built catalog);
    otherwise they are built from fetch_corpus. `engine_fn` defaults to the real
    #10 estimator via make_engine_provider; inject another for tests."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if jobs is None or specs is None:
        jobs, specs, catalog = build_emulable_jobs(
            max_studies=max_studies, datasets=tuple(datasets), sepsis_first=True,
            threshold=threshold, emulable_only=emulable_only, http_get=http_get)
    else:
        catalog = {"n_emulable": len(jobs), "n_sepsis_emulable": None,
                   "max_studies": max_studies, "datasets": list(datasets),
                   "note": "jobs injected"}

    engine_fn = engine_fn or make_engine_provider(covariates or [], adjustment=adjustment)
    drops = DropLog()
    run_kw = {"extract_fn": extract_fn, "engine_fn": engine_fn}
    if compare_fn is not None:
        run_kw["compare_fn"] = compare_fn
    n_written, drops = run_corpus_to_jsonl(
        jobs, list(datasets), out / "corpus.jsonl", drops=drops, **run_kw)

    comparisons = list(load_comparisons_jsonl(out / "corpus.jsonl"))
    analysis = _write_analysis(comparisons, specs, out, datasets=list(datasets),
                               context=context, figures=figures)

    (out / "drops.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in drops.items))

    summary = {
        "n_emulable_trials": len(jobs),
        "n_comparisons": n_written,
        "n_dropped": len(drops),
        "drops_by_reason": drops.by_reason(),
        "adjustment": adjustment,
        "datasets": list(datasets),
        "catalog": catalog,
        "out_dir": str(out),
        **analysis,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _synthetic_extract_fn(scale: int = 2):
    """Scaffold extract_fn (NOT real data): the #13 vignette's confounded 5-col
    stream, routed through the REAL cohort builder + REAL engine. Validates the
    driver end-to-end (incl. the crude-harm -> IPTW-benefit flip) until worker1's
    #101 loader lands. Swap `--synthetic` off to use the real loader."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import sepsis_vignette as vig

    return lambda plan, spec, dataset: vig.confounded_stream(scale)


def _real_extract_fn(datasets):
    """worker1's #101 MIMIC/eICU loader. Imported lazily so the synthetic scaffold
    runs without it; raises a clear pointer until #101 lands."""
    try:
        from .adapters.live_loader import make_extract_fn  # #101 (pending)
    except ImportError as exc:
        raise SystemExit(
            "Real MIMIC/eICU loader (#101) not available yet — run with --synthetic "
            "to validate the driver, or wait for worker1's loader. "
            f"(import error: {exc})")
    return make_extract_fn(datasets)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Live MIMIC/eICU corpus run (#102)")
    ap.add_argument("--out", default="live_outputs")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--max-studies", type=int, default=2000)
    ap.add_argument("--adjustment", default="iptw", help="iptw | psm | cox | crude")
    ap.add_argument("--synthetic", action="store_true",
                    help="scaffold: synthetic confounded stream (no real data) through the real engine")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    extract_fn = _synthetic_extract_fn() if a.synthetic else _real_extract_fn(a.datasets)
    summary = run_live(
        extract_fn=extract_fn, out_dir=a.out, datasets=tuple(a.datasets),
        max_studies=a.max_studies, adjustment=a.adjustment,
        context=not a.no_context, figures=not a.no_figures)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
