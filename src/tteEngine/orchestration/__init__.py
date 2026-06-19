"""orchestration: the resumable per-trial pipeline + batch runner (#12, tte1).

Generalizes emulaTTE core/pipeline.py + steps.py: one typed step per stage
(ctgov -> spec -> plan -> extract -> cohort -> TTE -> report) chained with
stop -> edit -> resume (check-&-correct + selective cache invalidation). Each
component call is a thin wrapper with a graceful fallback so a missing DB/engine
degrades instead of hard-failing the batch over trials x DBs. Stub — built via PR.
"""
