# Contributing — agent-driven build

This repository is built by a team of autonomous agents coordinating on the
`#tte` hub. Humans monitor progress through GitHub issues + PRs.

## Workflow conventions

- **Everything goes through GitHub.** Each unit of work is a numbered issue with
  a single owner (noted in the issue body). Implement it on a branch and open a
  PR that references the issue (`Closes #N`).
- **One branch per lane / issue.** `git checkout -b <lane>/<short-desc>`. Do not
  commit other lanes' work in your PR.
- **Code to the contracts, not to each other's internals.** The typed seams in
  `src/tteEngine/contracts/` are the integration boundaries; if you need to
  change one, flag it on `#tte` first (it affects every lane).
- **Match the surrounding style.** Minimal, technical, typed. Add deps via the
  appropriate `pyproject` extra.

## Architecture & lanes

```
ctgov trial
  -> TargetTrialSpec      (#2 probe)        contracts/trial_spec.py
  -> ExtractionPlan       SEAM (a)          contracts/extraction_plan.py
  -> Event stream (5-col) SEAM (b)          contracts/events.py   <- canonical store
  -> CohortResult         SEAM (c)          contracts/cohort.py
  -> TTEResult -> emulated-vs-observed report
```

| Lane | Owner | Issues |
|------|-------|--------|
| core: common-format + cohort + orchestration + vignette | tte1 | #0 #4 #5 #9 #12 #13 |
| extraction: per-DB adapters | worker1 | #6 #7 #8 |
| ctgov + analysis | probe | #1 #2 #3 #10 #11 |

The **common format** is the EHR-DE 5-column long event-stream
(`TRAJECTORY_ID, TIMESTAMP, EVENT_TYPE, EVENT_NAME, EVENT_VALUE`) — NOT OMOP.
Wide cohort/feature tables are deterministic **views** over this canonical
stream. Cross-DB semantics come from the vocab layer (#5), keeping structure and
semantics decoupled.

## Dev

```bash
pip install -e ".[dev,analysis,ctgov]"   # or: uv sync --all-extras
pytest
```
