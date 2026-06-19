"""tteEngine — trial-general, multi-dataset Target Trial Emulation.

Pipeline:  ctgov trial -> extraction plan -> per-DB extraction -> common EHR
format -> cohort -> TTE analysis -> emulated-vs-observed report.

Packages map 1:1 to the build lanes (see CONTRIBUTING.md / GitHub issues):
  contracts/       typed boundaries every lane codes to            (#4, tte1)
  common_format/   the 5-col canonical store + long->wide views    (#4, tte1)
  cohort/          eligibility/time-zero/arms over the stream      (#9, tte1)
  orchestration/   the resumable per-trial pipeline (the spine)    (#12, tte1)
  ctgov/           trial reader + NCT -> TargetTrialSpec + plan     (#1/#2/#3, probe)
  analysis/        TTE engine + emulated-vs-observed benchmark      (#10/#11, probe)
  adapters/        per-DB extraction to the common format          (#6/#7/#8, worker1)
  vocab/           EVENT_NAME/unit -> ICD/RxNorm/LOINC normalization (#5, worker1)
"""

__version__ = "0.0.1"
