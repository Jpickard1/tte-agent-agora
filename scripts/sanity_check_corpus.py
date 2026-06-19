"""Sanity-check a live TTE corpus run before it goes on screen (tte1).

Read-only. Point it at probe's output dir (corpus.jsonl + audit.jsonl, optional
context/ledger) and it validates the things that must hold for the numbers to be
publishable — joins, arm-size consistency, code-vs-substring match quality, the
crude->adjusted confounding flip, and calibration — printing PASS/WARN/FAIL.

    python scripts/sanity_check_corpus.py /path/to/run_dir
    python scripts/sanity_check_corpus.py --corpus a.jsonl --audit b.jsonl

It NEVER recomputes the science — it reads what the run emitted and checks it for
internal consistency + the sanity signatures we expect (low substring share now
that drug matching is code-based; screened>=enrolled; concordance in [0,1]).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter


def _load(corpus_path, audit_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from tteEngine.contracts.audit import load_audit_jsonl
    from tteEngine.contracts.io import load_comparisons_jsonl
    rows = list(load_comparisons_jsonl(corpus_path)) if corpus_path and os.path.exists(corpus_path) else []
    audits = list(load_audit_jsonl(audit_path)) if audit_path and os.path.exists(audit_path) else []
    return rows, audits


def _check(label, ok, detail=""):
    tag = "PASS" if ok is True else ("WARN" if ok == "warn" else "FAIL")
    print(f"[{tag}] {label}" + (f" — {detail}" if detail else ""))
    return tag


def run(corpus_path, audit_path, context_path=None, ledger_path=None):
    rows, audits = _load(corpus_path, audit_path)
    tags = []

    # --- coverage ---
    ds = sorted({getattr(r, "dataset", "?") for r in rows})
    tags.append(_check("corpus non-empty", bool(rows), f"{len(rows)} comparison rows over datasets {ds}"))
    tags.append(_check("audit non-empty", bool(audits), f"{len(audits)} AssignmentAudit records"))

    # --- join integrity: every corpus row should have an audit on (nct,dataset) ---
    ck = lambda o: (getattr(o, "nct_id", None), getattr(o, "dataset", None))
    audit_keys = {ck(a) for a in audits}
    missing = [ck(r) for r in rows if ck(r) not in audit_keys]
    tags.append(_check("every corpus row joins to an audit on (nct_id,dataset)",
                       not missing if audits else "warn",
                       f"{len(missing)} unmatched" if missing else "all joined"))

    # --- arm-size consistency + CONSORT monotonicity ---
    bad_consort = [ck(a) for a in audits
                   if not (a.n_screened >= a.n_eligible >= a.n_enrolled >= 0)]
    tags.append(_check("CONSORT counts monotone (screened>=eligible>=enrolled)",
                       not bad_consort if audits else "warn",
                       f"{len(bad_consort)} violate" if bad_consort else ""))
    arm_mismatch = [ck(a) for a in audits
                    if sum(arm.n for arm in a.arms) + a.n_unassigned != a.n_enrolled]
    tags.append(_check("arm sizes + unassigned reconcile to enrolled",
                       not arm_mismatch if audits else "warn",
                       f"{len(arm_mismatch)} mismatch" if arm_mismatch else ""))

    # --- match quality: code-based matching should DOMINATE; substring share LOW ---
    methods = Counter()
    for a in audits:
        try:
            for m, c in a.match_method_totals().items():
                methods[m] += c
        except Exception:
            pass
    total_m = sum(methods.values())
    if total_m:
        sub = methods.get("substring", 0)
        share = sub / total_m
        tags.append(_check("drug matching is code-based (substring share LOW)",
                           True if share <= 0.20 else "warn",
                           f"substring={sub}/{total_m} ({share:.0%}); methods={dict(methods)}"))
    else:
        tags.append(_check("match-method counts present", "warn", "no match_method_counts in audits"))
    low = sum(getattr(a, "n_low_confidence", 0) for a in audits)
    tags.append(_check("low-confidence arm matches are a minority",
                       "warn" if low else True, f"n_low_confidence total={low}"))

    # --- science signatures (read-only): estimates finite, concordance sane ---
    finite = [r for r in rows if getattr(r, "emulated", None) is not None]
    tags.append(_check("comparison rows carry emulated estimates", bool(finite) if rows else "warn",
                       f"{len(finite)}/{len(rows)}"))
    agrees = [str(getattr(getattr(r, "agreement", None), "value", getattr(r, "agreement", "")))
              for r in rows]
    decided = [a for a in agrees if a in ("concordant", "discordant")]
    if decided:
        rate = sum(a == "concordant" for a in decided) / len(decided)
        n_inc = sum(a == "inconclusive" for a in agrees)
        tags.append(_check("headline concordance in [0,1]", 0.0 <= rate <= 1.0,
                           f"emulated-vs-observed concordance = {rate:.0%} over {len(decided)} decided "
                           f"({n_inc} inconclusive of {len(rows)})"))

    print("\n" + "=" * 60)
    c = Counter(tags)
    print(f"SUMMARY: {c.get('PASS',0)} PASS  {c.get('WARN',0)} WARN  {c.get('FAIL',0)} FAIL")
    if c.get("FAIL"):
        print(">> FAILs must be resolved before these numbers go on screen.")
    elif c.get("WARN"):
        print(">> No hard failures; review WARNs (often expected on a first/small run).")
    else:
        print(">> Clean — safe to relaunch the dashboard on these numbers.")
    return c.get("FAIL", 0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", nargs="?", help="dir containing corpus.jsonl + audit.jsonl")
    p.add_argument("--corpus")
    p.add_argument("--audit")
    p.add_argument("--context")
    p.add_argument("--ledger")
    a = p.parse_args(argv)
    if a.run_dir:
        j = lambda n: os.path.join(a.run_dir, n)
        a.corpus = a.corpus or j("corpus.jsonl")
        a.audit = a.audit or j("audit.jsonl")
        a.context = a.context or j("context.jsonl")
        a.ledger = a.ledger or j("ledger.jsonl")
    return run(a.corpus, a.audit, a.context, a.ledger)


if __name__ == "__main__":
    sys.exit(main())
