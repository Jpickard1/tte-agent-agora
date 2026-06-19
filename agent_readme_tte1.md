# agent_readme — tte1 (core / orchestration / output / UI lane)

Handoff doc for agent **tte1**. Status as of the session pause. Everything in my
lane is **merged to `main`**; the one open item is the **canonical-run gallery
validation**, which is gated on @probe's in-flight run — steps to finish it are below.

## 1. Delivered (all merged on `main`)

| Area | What | PR(s) |
|---|---|---|
| Audit schema | `contracts/audit.py` — `Confidence` tiers, `MatchProvenance`, `EligibilityDecision`, `ArmAudit`, `AssignmentAudit`, `dump/load_audit_jsonl` | #140 #144 |
| Audit primitives | `build_cohort` emits `eligibility_decisions`, `assignment_provenance`, `arm_method_counts`, `n_unassigned`, `n_low_confidence`; `run_corpus(on_audit=...)` + `assignment_audit_from_cohort()` | #154 |
| Dataset binding | `run_corpus` binds the per-dataset `measurable_fn(crit, ds)` / `arm_match_fn(name, concepts, ds)` so a single resolver pair is correct across MIMIC+eICU | #156 |
| Outcome selection | `engine_provider` uses `select_measurable_outcome` + `outcome_column` (emulate the measurable endpoint, not blindly `outcomes[0]`) | #146 |
| Combo arm strategy (#162-A) | `Arm.strategy` (`any`/`all`); `_assign_with_prov` per-component coverage → `all` requires EVERY component co-administered in `(t0,t0+grace]`. Run default `all` (jpic-confirmed) set in `live_run.py`; library default conservative | #165 #170 |
| Outcome-safe t0 (#166) | `_index_times` falls back to earliest **non-outcome** event when the anchor is absent (never anchor on death → no immortal-exclude-all); outcome-only trajectories drop as `CohortDiagnostics.n_unanchorable` | #168 |
| Transparency UI (#130) | `ui/` dashboard + "HOW PATIENTS WERE SORTED" panels; `web/results_app.py` + `web/theme.py` (emulaTTE theme, ctgov LinkColumn) | #142 |
| Sanity gate | `scripts/sanity_check_corpus.py` — pre-publish read-only checks | #158 |

Seams I own (consumer-defined): `CohortResult`, `AssignmentAudit`, `ComparisonResult`,
`TTEResult`. Import-light layering holds: orchestration/figures/ui/contracts do NOT
import the heavy `analysis` extra.

## 2. IN-PROGRESS / PENDING TODO — canonical-run gallery validation

@probe's canonical run (PID 1158260, `arm_strategy='all'`, all fixes) writes to
**`/ewsc/jpickard/tte_live/canonical/`** (`corpus.jsonl`, `context.jsonl`,
`ledger.jsonl`, `audit.jsonl`, `summary.json`). When it lands:

**Step 1 — run the #158 sanity-gate** (set TMPDIR to dodge the shared-`/tmp` exit-144
contention flagged by probe):
```bash
TMPDIR=/ewsc/jpickard/tmp \
  python scripts/sanity_check_corpus.py /ewsc/jpickard/tte_live/canonical
```
Checks: corpus↔audit joins on `(nct_id,dataset)`, CONSORT monotonicity, arm +
`n_unassigned` reconcile to enrolled, **code-based-match share** (substring LOW =
#131 worked), concordance in `[0,1]`. If Bash output drops to 144, fall back to the
**Read tool** on the `/ewsc` jsonl files (reliable even when Bash drops).

**Step 2 — validate the per-protocol effect.** HAT trial (NCT03509350) under the
preliminary `any`-style run was Treatment=**4053** / Control=**17841** (control was 0
before the anchor fix). Under `arm_strategy='all'`, Treatment should **tighten** to
the true combo cohort (vit C + thiamine + hydrocortisone co-administered) **without
re-collapsing control to 0**. Confirm that in the audit arm sizes + `arm_method_counts`.

**Step 3 — relaunch :8520 on the run-dir** (read from `/ewsc`, never copy to `/home`):
```bash
export TTE_CORPUS_JSONL=/ewsc/jpickard/tte_live/canonical/corpus.jsonl
export TTE_CONTEXT_JSONL=/ewsc/jpickard/tte_live/canonical/context.jsonl
export TTE_LEDGER_JSONL=/ewsc/jpickard/tte_live/canonical/ledger.jsonl
export TTE_AUDIT_JSONL=/ewsc/jpickard/tte_live/canonical/audit.jsonl
streamlit run web/results_app.py   # needs the `web` extra
```
Then eyeball: real arm sizes, the sorting/provenance panel, the confounder ledger,
and the audit panel — all against the actual numbers. (@manager owns the relaunch;
tte1 clears the gate + eyeballs.)

## 3. Key decisions / gotchas for continuity
- **arm_strategy**: run default `'all'` lives in `live_run.py` (run_live + `--arm-strategy`);
  the `build_cohort`/`run_corpus` *library* defaults stay `None`/`Arm.strategy="any"`
  so unit tests + other callers are unopinionated. `'all'` is a no-op for single-component
  arms (n_required collapses to 1) — single-drug trials unaffected; control never forced.
- **Audit records, never gates**: emitting `EligibilityDecision`/`MatchProvenance` only
  *records* include/drop decisions — it must never change them.
- **Disk**: shared `/home` was at 92%. Write all big outputs to `/ewsc` (84T). Code
  worktrees stay on `/home` (small). `TMPDIR=/ewsc/jpickard/tmp` for big intermediates.
- **Dropped-commit race**: multi-commit PRs lost their 2nd commit to merge races 3×.
  ALWAYS `git --no-fetch grep -c <fix-marker> origin/main` after a merge; prefer
  squash-merge for multi-commit feature PRs. (Adopted as team process by @manager.)
- **hubcli**: `send <agent> --author tte1 "msg"` (positional body; avoid backticks —
  they get command-substituted).

## 4. Possible follow-ups (not started)
- Surface any new `MatchProvenance` fields worker1 adds (`source_table`, `dose`,
  `route`) in the #130 UI provenance render (field-tolerant).
- If the gallery shows multi-concept *alternative* arms (not combos) being wrongly
  required-all under the global `'all'` default, set per-arm `Arm.strategy='any'`
  on those specs (ctgov lane) — the mechanism already supports per-arm override.
