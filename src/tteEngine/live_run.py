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
from .contracts.audit import dump_audit_jsonl
from .contracts.io import load_comparisons_jsonl
from .orchestration.corpus import DropLog, run_corpus_to_jsonl
from .orchestration.engine_provider import make_engine_provider
from .triage import score_spec

DATASETS = ("MIMIC-IV", "eICU-CRD")


def _lean_plan_fn(spec, *, dataset):
    """Lean extraction plan: drop LAB/MEASUREMENT concepts, which trigger the
    multi-GB per-trial labevents/chartevents scans (and mostly back outcomes that
    aren't measurable in ICU EHR anyway). Keeps the cohort dx + medications (arm
    assignment) + mortality outcome -> seconds/trial instead of minutes, which is
    what makes the >=1k corpus feasible. Use lean=False for full extraction (labs
    as covariates) on a small corpus or once a shared pre-pass exists."""
    from .contracts.events import EventType
    from .ctgov import spec_to_plan

    plan = spec_to_plan(spec, dataset=dataset)
    plan.concepts = [c for c in plan.concepts
                     if c.event_type not in (EventType.LAB, EventType.MEASUREMENT)]
    return plan


def build_emulable_jobs(*, max_studies, datasets=DATASETS, sepsis_first=True,
                        threshold=0.5, emulable_only=True, http_get=None):
    """Fetch the sepsis-first ctgov catalog and keep trials emulable in >=1 dataset.

    Returns (jobs, specs, catalog). jobs = [(study_dict, TargetTrialSpec)] for
    run_corpus. `catalog` reports n_fetched / n_emulable / n_unemulable / n_sepsis
    and the explicit max_studies cap — coverage is never silently overstated."""
    from .ctgov import fetch_corpus, study_to_spec

    kw = {"http_get": http_get} if http_get is not None else {}
    studies = fetch_corpus(max_studies=max_studies, sepsis_first=sepsis_first, **kw)
    tagged: list[tuple[bool, dict, object]] = []  # (is_sepsis, study, spec)
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
        is_sep = bool(scores and scores[0].is_sepsis)
        n_sepsis += is_sep
        tagged.append((is_sep, s, spec))
    # SEPSIS-FIRST within the emulable set (stable): sepsis trials processed/persisted
    # first, so a partial run still has the priority cohort + the gallery leads with it.
    tagged.sort(key=lambda t: not t[0])
    jobs = [(s, spec) for _, s, spec in tagged]
    specs = [spec for _, _, spec in tagged]
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

    # #105 confounder adjustability ledger + PS diagnostics, joined on (nct_id,dataset)
    from .adjustability import write_ledger_sidecar
    n_ledger = write_ledger_sidecar(comparisons, specs, out / "ledger.jsonl")

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
        "n_ledger": n_ledger,
        "forest_figure": figure_path,
    }


def run_live(*, extract_fn, engine_fn=None, compare_fn=None, plan_fn=None, jobs=None, specs=None,
             measurable_fn=None, arm_match_fn=None,
             out_dir="live_outputs", datasets=DATASETS, max_studies=2000, max_trials=None,
             adjustment="iptw", covariates=None, threshold=0.5, emulable_only=True,
             arm_strategy="all", lean=True, http_get=None, context=True, figures=True,
             audit=True) -> dict:
    """Run the live corpus end-to-end and write the gallery artifacts to out_dir:
    corpus.jsonl, context.jsonl, ledger.jsonl, RESULTS_NARRATIVE.md, drops.jsonl,
    summary.json (+ forest.png). `jobs`/`specs` may be injected (tests / a pre-built
    catalog); otherwise built from fetch_corpus. `engine_fn` defaults to the real #10
    estimator. lean=True (default) prunes the multi-GB lab/measurement scans so a
    >=1k-trial corpus is feasible; pass plan_fn to override the plan entirely."""
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

    if max_trials is not None and len(jobs) > max_trials:
        catalog["n_capped_to"] = max_trials      # explicit cap (sepsis-first), never silent
        jobs, specs = jobs[:max_trials], specs[:max_trials]

    engine_fn = engine_fn or make_engine_provider(covariates or [], adjustment=adjustment)
    if plan_fn is None and lean:
        plan_fn = _lean_plan_fn
    drops = DropLog()
    audits: list = []
    run_kw = {"extract_fn": extract_fn, "engine_fn": engine_fn}
    if compare_fn is not None:
        run_kw["compare_fn"] = compare_fn
    if plan_fn is not None:
        run_kw["plan_fn"] = plan_fn
    # #131 code-correct matching + #138 measurability-aware eligibility at the cohort seam
    if measurable_fn is not None:
        run_kw["measurable_fn"] = measurable_fn
    if arm_match_fn is not None:
        run_kw["arm_match_fn"] = arm_match_fn
    # #162: per-protocol combo arms (jpic-confirmed default 'all') — a combo trial is
    # 'treated' only if EVERY component is co-administered in-window, so matching one
    # routine banana-bag component (thiamine) no longer over-includes. 'all' is a no-op
    # for single-component arms (n_required collapses to 1), so single-drug trials are
    # unchanged; control arms are never forced.
    if arm_strategy is not None:
        run_kw["arm_strategy"] = arm_strategy
    # #143/#130: collect the per-(nct,dataset) AssignmentAudit (tte1 assembles it via
    # on_audit) so we can persist audit.jsonl for the 'how patients were sorted' panel
    if audit:
        run_kw["on_audit"] = audits.append
    n_written, drops = run_corpus_to_jsonl(
        jobs, list(datasets), out / "corpus.jsonl", drops=drops, **run_kw)

    comparisons = list(load_comparisons_jsonl(out / "corpus.jsonl"))
    analysis = _write_analysis(comparisons, specs, out, datasets=list(datasets),
                               context=context, figures=figures)

    n_audit = dump_audit_jsonl(audits, out / "audit.jsonl") if audit else 0

    (out / "drops.jsonl").write_text(
        "".join(json.dumps(d) + "\n" for d in drops.items))

    summary = {
        "n_emulable_trials": len(jobs),
        "n_comparisons": n_written,
        "n_dropped": len(drops),
        "drops_by_reason": drops.by_reason(),
        "adjustment": adjustment,
        "arm_strategy": arm_strategy,
        "lean": bool(plan_fn is _lean_plan_fn),
        "n_audit": n_audit,
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
    ap.add_argument("--max-trials", type=int, default=None,
                    help="cap the number of (sepsis-first) trials run (e.g. for a quick real slice)")
    ap.add_argument("--adjustment", default="iptw", help="iptw | psm | cox | crude")
    ap.add_argument("--arm-strategy", default="all", choices=["all", "any"],
                    help="combo arm matching: 'all' = per-protocol (jpic-confirmed default, "
                         "every component required); 'any' = treated if any component present")
    ap.add_argument("--synthetic", action="store_true",
                    help="scaffold: synthetic confounded stream (no real data) through the real engine")
    ap.add_argument("--full", action="store_true",
                    help="full extraction (lab/measurement covariates); default is LEAN (#124) for scale")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)

    extract_fn = _synthetic_extract_fn() if a.synthetic else _real_extract_fn(a.datasets)
    summary = run_live(
        extract_fn=extract_fn, out_dir=a.out, datasets=tuple(a.datasets),
        max_studies=a.max_studies, max_trials=a.max_trials, adjustment=a.adjustment,
        arm_strategy=a.arm_strategy,
        lean=not a.full, context=not a.no_context, figures=not a.no_figures)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
