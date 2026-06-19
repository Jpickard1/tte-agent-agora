"""#35 corpus runner — fetch a large ctgov corpus and emit the ranked emulability
catalog (CSV + JSON + summary).

This is the orchestration over probe's corpus bridge (#1->#58) + the scorer (#35):

    fetch_corpus(sepsis_first=True, max_studies=N)  # COMPLETED + results-posted
        -> study_to_spec(s)                          # ctgov -> TargetTrialSpec (#2)
        -> build_catalog(specs)                      # per-dataset emulability (#35)
        -> catalog.csv + catalog.json + summary.json

jpic's directive (verbatim): MAXIMIZE count — >1000 minimum, >10k excellent; if
restricting, PRIORITIZE sepsis trials; NEVER silently cap — log what's dropped +
why. So this runner:
  * keeps EVERY trial in the catalog (low-emulability is flagged, not dropped);
  * records studies that fail to parse to a spec (never drops silently — they go
    in summary['unparseable'] + a logged warning);
  * when the configured `max_studies` cap is hit, that is logged as an EXPLICIT
    cap (not silent) in summary['cap'] so a larger corpus is an obvious next run.

Pure stdlib (csv/json/logging) — no pandas, so it runs in CI's [dev] env.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from tteEngine.ctgov import fetch_corpus, study_to_spec
from tteEngine.ctgov.reader import DEFAULT_CACHE, nct_id_of

from . import DATASETS, build_catalog

log = logging.getLogger("tteEngine.triage.runner")

#: column order for catalog.csv (one row per trial x dataset).
CSV_COLUMNS = [
    "nct_id", "dataset", "is_sepsis", "emulable", "score",
    "eligibility_ok", "exposure_ok", "outcome_ok", "reasons",
]


def _specs_from_studies(studies: list[dict]) -> tuple[list, list[dict]]:
    """Parse each study -> TargetTrialSpec. Studies that fail to parse are NOT
    dropped silently: they are collected (nct + error) and logged."""
    specs, unparseable = [], []
    for s in studies:
        nct = "?"
        try:
            nct = nct_id_of(s) or "?"
            specs.append(study_to_spec(s))
        except Exception as exc:  # never let one bad study sink the corpus run
            unparseable.append({"nct_id": nct, "error": f"{type(exc).__name__}: {exc}"})
            log.warning("triage: could not parse a study to a spec: %s", exc)
    if unparseable:
        log.warning("triage: %d/%d studies unparseable (kept in summary, not dropped)",
                    len(unparseable), len(studies))
    return specs, unparseable


def run_corpus_triage(
    *,
    max_studies: int = 2000,
    sepsis_first: bool = True,
    datasets: tuple[str, ...] = DATASETS,
    threshold: float = 0.5,
    out_dir: str | Path | None = None,
    studies: list[dict] | None = None,
    http_get=None,
    cache_dir=DEFAULT_CACHE,
) -> dict:
    """Build the ranked emulability catalog over a ctgov corpus.

    Returns the catalog dict (``{"catalog": [...], "summary": {...}}``) with the
    summary extended by corpus provenance (fetched/unparseable counts, the
    explicit cap flag, sepsis-first). When ``out_dir`` is given, writes
    ``catalog.csv``, ``catalog.json`` and ``summary.json`` there and records the
    paths in ``summary['outputs']``.

    `studies` (optional) bypasses the live fetch (used by tests / offline runs);
    otherwise `fetch_corpus` is called with `http_get`/`cache_dir` (inject a stub
    `http_get` for hermetic CI).
    """
    if studies is None:
        studies = fetch_corpus(max_studies=max_studies, sepsis_first=sepsis_first,
                               cache_dir=cache_dir, http_get=http_get)
    log.info("triage: %d studies in corpus (sepsis_first=%s, cap=%d)",
             len(studies), sepsis_first, max_studies)

    specs, unparseable = _specs_from_studies(studies)
    catalog = build_catalog(specs, datasets=datasets, threshold=threshold)

    # explicit (logged, never silent) cap flag: did we hit the configured ceiling?
    capped = len(studies) >= max_studies
    if capped:
        log.warning("triage: corpus hit the configured cap of %d studies — the true "
                    "emulable set may be larger; re-run with a higher max_studies.", max_studies)
    catalog["summary"].update({
        "n_studies_fetched": len(studies),
        "n_specs_parsed": len(specs),
        "n_unparseable": len(unparseable),
        "unparseable": unparseable,
        "sepsis_first": sepsis_first,
        "cap": {"max_studies": max_studies, "hit": capped},
    })

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = catalog["catalog"]
        with (out / "catalog.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({**r, "reasons": " | ".join(r.get("reasons", []))})
        (out / "catalog.json").write_text(json.dumps(rows, indent=2))
        (out / "summary.json").write_text(json.dumps(catalog["summary"], indent=2))
        catalog["summary"]["outputs"] = {
            "catalog_csv": str(out / "catalog.csv"),
            "catalog_json": str(out / "catalog.json"),
            "summary_json": str(out / "summary.json"),
        }
        log.info("triage: wrote catalog (%d rows) + summary to %s", len(rows), out)

    return catalog


__all__ = ["run_corpus_triage", "CSV_COLUMNS"]
