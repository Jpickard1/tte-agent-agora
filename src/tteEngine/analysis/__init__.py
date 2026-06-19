"""analysis: estimand-aware TTE engine (#10) + emulated-vs-observed benchmark (#11).

#10 wraps trialsim app/trialsim/tte_engine.py (PSM/IPTW/Cox/AIPW/KM/log-rank/
E-values/robustness/replication); consumes contracts.CohortResult, returns a
TTEResult. #11 reuses trialsim compare.py (ctgov resultsSection -> RR/RD +
concordance) and adds the batch harness over trials x DBs. Owner: probe. Stub.
"""
