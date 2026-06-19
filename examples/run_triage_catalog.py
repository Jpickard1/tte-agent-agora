"""#35 CLI: build the ranked emulability catalog over a live ctgov corpus.

    python examples/run_triage_catalog.py --max-studies 2000 --out-dir ./catalog

Sepsis-first, COMPLETED + results-posted, paged/cached (offline-replayable after
the first run). NEVER silently caps: every trial lands in catalog.csv/json with
its score + reasons, and summary.json logs the drop-reason distribution + an
explicit cap flag. jpic's target: >1000 minimum, >10k excellent.
"""
import argparse
import json
import logging

from tteEngine.triage import run_corpus_triage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-studies", type=int, default=2000,
                    help="corpus ceiling (>=1000 recommended; >10k = excellent)")
    ap.add_argument("--out-dir", default="./catalog", help="where to write catalog.{csv,json} + summary.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-sepsis-first", action="store_true", help="disable sepsis prioritization")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cat = run_corpus_triage(
        max_studies=args.max_studies, sepsis_first=not args.no_sepsis_first,
        threshold=args.threshold, out_dir=args.out_dir,
    )
    print(json.dumps(cat["summary"], indent=2))


if __name__ == "__main__":
    main()
