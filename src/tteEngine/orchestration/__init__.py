"""orchestration: the resumable per-trial pipeline + batch runner (#12, tte1).

Generalizes emulaTTE core/pipeline.py + steps.py: one typed step per stage
(ctgov -> spec -> plan -> extract -> cohort -> TTE -> report) chained with
stop -> edit -> resume (check-&-correct + selective cache invalidation). Each
component call is a thin wrapper with a graceful fallback so a missing DB/engine
degrades instead of hard-failing the batch over trials x DBs.
"""

from .pipeline import STEPS, Pipeline, TargetRequest, default_providers

__all__ = ["Pipeline", "TargetRequest", "STEPS", "default_providers"]
