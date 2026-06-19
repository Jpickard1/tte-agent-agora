# Probe — agent status & resume notes (live-run phase, 2026-06-19)

Lane: ctgov ingestion/spec/intelligence (#1/#2/#3) · engine/benchmark (#10/#11) ·
analysis/journal (#37/#38/#39/#41/#61/#63/#64) · reproducibility (#62) · ARCH doc
(#14) · **live-run driver (#102)** · **confounder ledger (#105)** ·
**outcome-selection (#146)** · **audit persistence (#143)**.

## DELIVERED + MERGED (probe)
`#102` live-run driver · `#145` lean mode + sepsis-first + max-trials · `#105`
confounder/PS adjustability ledger · `#120` public measure_fn · `#146`
select_measurable_outcome + like-for-like compare-alignment · `#143` audit.jsonl
persistence. All on `main`.

## IN PROGRESS — #111 CANONICAL SEPSIS-FIRST RUN (the gallery numbers)

**State: METHODS PROVEN; executing the proof run.** Every methods blocker the real
MIMIC data exposed is fixed + verified on `main`:
- `#161` drug-matcher (Drug: prefix + ingredient synonyms; was 0 codes → real codes)
- `#163/#164` eligibility skip-unmeasurable (don't fail-all on pruned criteria)
- `#167` adapter emits LOCATION `icu_admission` anchor + `#168` build_cohort
  outcome-safe t0 fallback — **THE anchor/immortal-time fix**
- `#122/#169` emulability down-ranks device/non-drug + routine-only (banana-bag) drugs
- `#165` combo-A arm matching; `arm_strategy='all'` default via `#170` (jpic-confirmed)
- `#146` outcome-selection (emulate the measurable mortality endpoint) + compare-alignment

**PROVEN via per-trial characterizer on main:** cohorts populate (n_total ~21,894),
**control arms FILL** (HAT NCT03509350: Treatment 4053 / Control 17841 — was 0
control before the anchor). Control-only trials (drug not code-matched) drop
missing-treatment-arm (correct); device/outcome-only down-ranked out. The earlier
0/24 is GONE — the anchor is the unlock.

### HOW TO RUN IT (critical — these are hard-won)
```
# MUST use setsid (own session/process group) — plain nohup gets REAPED externally
# (a contending harness/tmp-cleaner process-group-kills nohup runs mid-trial-0).
setsid python -u /ewsc/jpickard/tte_live/instr_run.py 24 \
  </dev/null >/ewsc/jpickard/tte_live/instr.log 2>&1 &
# verify it survives: pgrep -f instr_run.py ; sid==pid means session leader (good)
```
- `instr_run.py` (staged in /ewsc/jpickard/tte_live/) = per-trial-logged canonical run
  (fetch sepsis-first → emulable jobs → run_corpus(extract+cohort+IPTW, arm_strategy='all',
  on_audit) → corpus/audit/drops). Self-contained (embeds sys.path). Logs OK/DROP per trial.
- `run_canonical.py` = the run_live wrapper (same output); use either.
- **Output dir = `/ewsc/jpickard/tte_live/canonical`** (corpus/context/ledger/audit.jsonl).
  NEVER write big outputs to `/home` (shared 20G fs, 92% full) — use `/ewsc` (84T).
  Set `TMPDIR=/ewsc/jpickard/tmp`.
- **Do NOT bind measurable_fn** = eligibility_measurable — it OVER-ENFORCES the
  age/demographic + dx criteria → 0 eligible. build_cohort's default (enforce only
  criteria whose event_type is present in the stream) is correct. (Verified: binding
  it → 0/8 even no-landmark.)
- `arm_strategy='all'` (per-protocol combo, jpic-confirmed) = run_live default on main.

### PERFORMANCE / SCALE
Each trial does a full ~579MB `prescriptions.csv.gz` read for drug-code matching
(`live_loader.py:159`) — ~1-2 min/trial. N=24 ≈ 30-40 min; **FATAL for #111 ≥1k
(~16-33h).** **worker1's `#171` drug-prepass** (scan prescriptions ONCE, filtered to
the corpus-union drug codes, cache to parquet, slice per cohort) is the scaling fix —
ADOPT it for the scaled run.

### YIELD REALITY (honest)
Many sepsis "trials" are device/monitor trials (TORAYMYXIN, Starling SV → no drug →
not emulable, down-ranked #122) or drugs not in MIMIC. So per random slice only a few
trials yield both arms + a measurable outcome. For jpic's ≥1k target, fetch a LARGE
catalog (#171 makes that feasible) and accept the emulable fraction.

### AFTER THE RUN COMPLETES (the handoff)
1. Read `/ewsc/jpickard/tte_live/canonical/summary.json` → populated count, concordance,
   calibration, i2; verify `audit.jsonl` method-counts show CODE matching (not substring).
2. Post the run-dir + the 4 sidecars to #tte → **tte1 runs the `#158` sanity-gate**
   (`scripts/sanity_check_corpus.py <run_dir>`) → **manager relaunches :8520** + pings jpic.

## INFRA LESSONS (this session)
- **setsid** to detach long runs (nohup alone gets reaped here).
- Write run outputs to **/ewsc** (/home full); observe via the **Read tool** (the harness
  /tmp task-output fs fills on long sessions → Bash 144s; Read bypasses it).
- "merged" ≠ "works on real data" — verify each resolver/fix on a REAL ctgov spec +
  REAL MIMIC extract before trusting it (caught: build_drug_matcher→0 codes; the anchor
  bug; routine-drug over-match; the compare-alignment dropped-commit).

## REMAINING TODOs (probe)
- [ ] Let the in-flight canonical run (PID 1242878, setsid) finish → report + gallery.
- [ ] Adopt worker1's `#171` drug-prepass → re-run for the scaled `#111` (≥1k, both DBs).
- [ ] Verify audit method-counts are code-based (rxnorm_code/icd_hierarchy >> substring)
      once a run completes; tie pruned-for-scale lab confounders honestly in the #105 ledger.
- [ ] eICU arm of the corpus once its drug/dx code coverage is validated.
