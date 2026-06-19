"""Cohort builder over the canonical 5-col stream (#9, tte1).

Reads the canonical long event-stream + a TargetTrialSpec and produces:
  - eligibility-filtered trajectories,
  - a LANDMARK time-zero per trajectory (grace window -> immortal-time safe),
  - treatment-strategy arm assignment,
  - and an analysis-ready WIDE frame (deterministic view via materialize_wide).

Emits contracts.CohortResult — the #9->#10 seam the TTE engine consumes.

v1 scope: the eligibility/arm/time-zero machinery is general and tested on
synthetic streams. Concept->event_type resolution for free-text covariates is
deferred to the vocab layer (#5) + the ExtractionPlan (#3); until then a
covariate is matched by EVENT_NAME with a caller-supplied event_type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tteEngine.common_format import Aggregation, FeatureSpec, materialize_wide, validate_canonical
from tteEngine.contracts.cohort import ArmAssignment, CohortDiagnostics, CohortResult
from tteEngine.contracts.events import EventType
from tteEngine.contracts.trial_spec import Comparator, EligibilityCriterion, TargetTrialSpec

if TYPE_CHECKING:
    import pandas as pd

_NUMERIC_CMP = {Comparator.GT, Comparator.GE, Comparator.LT, Comparator.LE, Comparator.EQ}


def _identity(name):
    return name


def _index_times(events: "pd.DataFrame", spec: TargetTrialSpec) -> dict[int, "pd.Timestamp"]:
    """Landmark t0 per trajectory: first event matching the anchor, else the
    trajectory's earliest event. anchor 'icu_admission' maps to LOCATION events.
    """
    anchor = spec.time_zero.anchor
    t0: dict[int, "pd.Timestamp"] = {}
    for tid, g in events.groupby("TRAJECTORY_ID", sort=True):
        g = g.sort_values("TIMESTAMP")
        hit = g
        if anchor == "icu_admission":
            loc = g[g["EVENT_TYPE"] == EventType.LOCATION.value]
            hit = loc if len(loc) else g
        else:
            named = g[g["EVENT_NAME"] == anchor]
            hit = named if len(named) else g
        t0[int(tid)] = hit["TIMESTAMP"].iloc[0]
    return t0


def _window_mask(sub: "pd.DataFrame", t0, window_hours) -> "pd.DataFrame":
    if window_hours is None:
        return sub
    lo, hi = window_hours
    rel = (sub["TIMESTAMP"] - t0).dt.total_seconds() / 3600.0
    return sub[(rel >= lo) & (rel <= hi)]


def _criterion_satisfied(traj_events: "pd.DataFrame", crit: EligibilityCriterion, t0, resolve) -> bool:
    import pandas as pd

    sub = traj_events[traj_events["EVENT_TYPE"] == crit.event_type.value]
    if crit.concept is not None:
        # match raw EVENT_NAME==concept OR resolve(EVENT_NAME)==concept, so both
        # concept-name streams (synthetic) and raw-coded adapter streams (#5<->#9) work.
        names = sub["EVENT_NAME"]
        mask = names == crit.concept
        if resolve is not _identity:  # skip the per-row map under the identity default (#36 scale)
            mask = mask | (names.map(resolve) == crit.concept)
        sub = sub[mask]
    sub = _window_mask(sub, t0, crit.window_hours)
    if len(sub) == 0:
        return False
    if crit.comparator == Comparator.EXISTS:
        return True
    if crit.comparator == Comparator.IN:
        allowed = crit.value if isinstance(crit.value, list) else [crit.value]
        return bool(sub["EVENT_VALUE"].isin([str(a) for a in allowed]).any())
    # numeric comparators: any matching event satisfies
    nums = pd.to_numeric(sub["EVENT_VALUE"], errors="coerce").dropna()
    if len(nums) == 0 or crit.value is None:
        return False
    v = float(crit.value)  # type: ignore[arg-type]
    if crit.comparator == Comparator.GT:
        return bool((nums > v).any())
    if crit.comparator == Comparator.GE:
        return bool((nums >= v).any())
    if crit.comparator == Comparator.LT:
        return bool((nums < v).any())
    if crit.comparator == Comparator.LE:
        return bool((nums <= v).any())
    if crit.comparator == Comparator.EQ:
        return bool((nums == v).any())
    return False


def _is_eligible(traj_events: "pd.DataFrame", criteria, t0, resolve) -> bool:
    """Apply only the (measurable) `criteria` — unmeasurable ones are dropped
    upstream and reported, so an un-emitted criterion (e.g. demographics) doesn't
    silently fail the whole cohort."""
    for crit in criteria:
        satisfied = _criterion_satisfied(traj_events, crit, t0, resolve)
        if crit.include and not satisfied:
            return False
        if (not crit.include) and satisfied:  # exclusion criterion triggered
            return False
    return True


def _norm_drug(name) -> str:
    """Normalize an intervention/med name for matching: drop the ctgov type prefix
    ('Drug:'/'Device:'/'Biological:'/...) and lowercase. 'Drug: Thiamine' -> 'thiamine'."""
    s = str(name)
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.strip().lower()


def _assign_arm(traj_events: "pd.DataFrame", spec: TargetTrialSpec, t0, resolve) -> str:
    """First treatment arm whose intervention is administered within the grace
    window wins; otherwise control. Matches exact / resolved name OR a normalized
    substring ('Drug: Thiamine' vs MIMIC 'thiamine'/'thiamine 100mg')."""
    grace = spec.time_zero.grace_window_hours
    treatment_arms = [a for a in spec.arms if not a.is_control]
    control = next((a for a in spec.arms if a.is_control), None)
    is_med = traj_events["EVENT_TYPE"] == EventType.MEDICATION.value
    names = traj_events["EVENT_NAME"]
    for arm in treatment_arms:
        concepts_norm = [c for c in (_norm_drug(x) for x in arm.intervention_concepts) if len(c) >= 3]

        def _norm_match(name, _cn=concepts_norm):
            n = _norm_drug(name)
            return any(cn in n or n in cn for cn in _cn)

        match = names.isin(arm.intervention_concepts) | names.map(_norm_match)
        if resolve is not _identity:  # skip the per-row map under the identity default (#36 scale)
            match = match | names.map(resolve).isin(arm.intervention_concepts)
        meds = traj_events[is_med & match]
        meds = _window_mask(meds, t0, (0.0, grace))
        if len(meds) > 0:
            return arm.name
    return control.name if control else "control"


def _outcome_before_landmark(traj_events, spec, t0, landmark_hours, resolve) -> bool:
    """True if a trial outcome occurs in [t0, t0+landmark): the patient leaves
    before reaching the landmark, so including them would create immortal time."""
    for outcome in spec.outcomes:
        sub = traj_events[traj_events["EVENT_TYPE"] == outcome.event_type.value]
        if outcome.concept is not None:
            names = sub["EVENT_NAME"]
            mask = names == outcome.concept
            if resolve is not _identity:
                mask = mask | (names.map(resolve) == outcome.concept)
            sub = sub[mask]
        if len(sub) == 0:
            continue
        rel = (sub["TIMESTAMP"] - t0).dt.total_seconds() / 3600.0
        if bool(((rel >= 0) & (rel < landmark_hours)).any()):
            return True
    return False


def _leakage_warnings(spec) -> list[str]:
    """Flag eligibility criteria assessed with POST-t0 data (window upper bound
    > 0) -- a classic look-ahead / immortal-time leak; eligibility should be
    determinable at or before time-zero."""
    warns = []
    for c in spec.eligibility:
        w = c.window_hours
        if w is not None and w[1] > 0:
            warns.append(
                f"eligibility '{c.concept or c.event_type.value}' uses a post-t0 window "
                f"(up to +{w[1]}h): assess at/before t0 to avoid look-ahead leakage."
            )
    return warns


def _component_match(name, concept, resolve, arm_match_fn):
    """Does med `name` match a SINGLE intervention component `concept`?
    -> (ok, code, method). With arm_match_fn (worker1's code matcher) the concept is
    matched by code; the default is exact/resolved-name then normalized-substring (LOW)."""
    if arm_match_fn is not None:
        res = arm_match_fn(name, [concept])  # one-concept set -> per-component coverage
        return bool(res[0]), (res[1] if len(res) > 1 else None), (res[2] if len(res) > 2 else "name")
    n, cn = _norm_drug(name), _norm_drug(concept)
    if "placebo" in n:
        return False, None, "substring"
    if name == concept or n == cn:
        return True, None, "name"                         # exact (raw or normalized)
    if resolve is not _identity and resolve(name) == concept:
        return True, None, "name"                         # vocab-resolved (#5)
    if len(cn) >= 3 and (cn in n or n in cn):
        return True, None, "substring"                    # last-resort substring (LOW)
    return False, None, "substring"


def _assign_with_prov(traj_events, spec, t0, resolve, arm_match_fn, arm_strategy=None):
    """Return (arm_name, provenance|None). Assigns to the first treatment arm the
    trajectory satisfies under that arm's TREATMENT STRATEGY (#162):

      * 'any'  -> treated if ANY intervention component is administered in-window
                  (single-drug or alternative-of arms);
      * 'all'  -> PER-PROTOCOL combination: treated only if EVERY component is
                  co-administered in (t0, t0+grace] (e.g. HAT = vit C + thiamine +
                  hydrocortisone). Matching one routine component (banana-bag thiamine)
                  no longer flips a patient to 'treated', so the control arm survives.

    `arm_strategy` overrides every arm's own `strategy` when given (so a run can force
    per-protocol matching across a combo-heavy corpus). provenance is a MatchProvenance-
    shaped dict for the protocol-defining event; for 'all' it marks protocol COMPLETION
    (the last required component) and reports the weakest match tier across components."""
    grace = spec.time_zero.grace_window_hours
    treatment_arms = [a for a in spec.arms if not a.is_control]
    control = next((a for a in spec.arms if a.is_control), None)
    meds = traj_events[traj_events["EVENT_TYPE"] == EventType.MEDICATION.value]
    meds = _window_mask(meds, t0, (0.0, grace)).sort_values("TIMESTAMP")
    for arm in treatment_arms:
        concepts = arm.intervention_concepts
        strat = (arm_strategy or getattr(arm, "strategy", "any") or "any").lower()
        # earliest in-window match per component -> concept: (t_rel, code, method, name)
        comp: dict[str, tuple] = {}
        for _, row in meds.iterrows():
            name = row["EVENT_NAME"]
            for c in concepts:
                if c in comp:
                    continue
                ok, code, method = _component_match(name, c, resolve, arm_match_fn)
                if ok:
                    t_rel = (row["TIMESTAMP"] - t0).total_seconds() / 3600.0
                    comp[c] = (round(float(t_rel), 2), code, method, str(name))
        n_required = len(concepts) if (strat == "all" and concepts) else 1
        if comp and len(comp) >= n_required:
            # 'all' -> protocol completes at the LAST required component; 'any' -> first exposure
            pick = max if strat == "all" else min
            rep_c = pick(comp, key=lambda k: comp[k][0])
            t_rel, code, method, ename = comp[rep_c]
            if any(v[2] == "substring" for v in comp.values()):  # audit honesty: weakest tier
                method = "substring"
            return arm.name, {"arm": arm.name, "matched_event_name": ename,
                              "matched_code": code, "concept": rep_c, "method": method,
                              "t_rel_hours": t_rel}
    return (control.name if control else "control"), None


def build_cohort(
    events: "pd.DataFrame",
    spec: TargetTrialSpec,
    *,
    dataset: str,
    validate: bool = True,
    resolve=None,
    enforce_landmark: bool = True,
    landmark_hours: float | None = None,
    skip_unmeasurable: bool = True,
    measurable_fn=None,
    arm_match_fn=None,
    arm_strategy: str | None = None,
    provenance_sample: int = 50,
) -> CohortResult:
    """Build the emulated-trial cohort with explicit, immortal-time-safe time-zero.

    t0 = the landmark anchor (spec.time_zero.anchor); treatment status is assessed
    over the grace window (t0, t0+grace], and follow-up conceptually starts at the
    landmark t0+`landmark_hours` (default = grace). IMMORTAL-TIME GUARD (#30): when
    `enforce_landmark`, trajectories whose trial outcome occurs before the landmark
    are EXCLUDED (they could not have survived to be assigned) and counted in
    diagnostics -- never silent. Post-t0 eligibility windows are flagged too.

    `resolve` (#5<->#9): EVENT_NAME->concept mapping for raw-coded streams; default
    identity leaves concept-name streams unchanged.
    """
    if validate:
        validate_canonical(events)
    _resolve = resolve or _identity
    landmark = landmark_hours if landmark_hours is not None else spec.time_zero.grace_window_hours

    # measurability-aware eligibility: drop criteria this dataset can't measure
    # (e.g. demographics not emitted) rather than failing every trajectory.
    present_types = set(events["EVENT_TYPE"].unique())

    def _measurable(crit):
        """-> (measurable: bool, reason: str|None). Accepts measurable_fn returning
        bool OR (bool, reason) (worker1's eligibility_measurable shape)."""
        if measurable_fn is not None:
            r = measurable_fn(crit)
            if isinstance(r, tuple):
                return bool(r[0]), (r[1] if len(r) > 1 else None)
            return bool(r), None
        present = crit.event_type.value in present_types
        return present, (None if present else "event-type not present in dataset")

    # per-criterion eligibility audit (#138): applied (enforced) vs skipped-unmeasurable + why
    applied, skipped_labels, eligibility_decisions = [], [], []
    for c in spec.eligibility:
        meas, reason = _measurable(c)
        enforce = meas or not skip_unmeasurable
        if enforce:
            applied.append(c)
        else:
            skipped_labels.append(f"{c.concept or c.event_type.value} ({c.event_type.value})")
        eligibility_decisions.append({
            "concept": c.concept, "event_type": c.event_type.value,
            "comparator": getattr(c.comparator, "value", str(c.comparator)),
            "value": c.value, "measurable": meas,
            "result": "applied" if enforce else "skipped_unmeasurable", "reason": reason,
        })

    t0_all = _index_times(events, spec)
    arms: dict[str, list[int]] = {}
    index_times: dict[int, object] = {}
    by_id = {int(tid): g for tid, g in events.groupby("TRAJECTORY_ID", sort=True)}

    n_eligible = n_excluded_immortal = n_unassigned = n_low_conf = 0
    provenance: list[dict] = []
    arm_method_counts: dict[str, dict[str, int]] = {}
    treatment_names = {a.name for a in spec.arms if not a.is_control}
    for tid in sorted(by_id):
        t0 = t0_all[tid]
        traj = by_id[tid]
        if not _is_eligible(traj, applied, t0, _resolve):
            continue
        n_eligible += 1
        if enforce_landmark and _outcome_before_landmark(traj, spec, t0, landmark, _resolve):
            n_excluded_immortal += 1
            continue
        arm_name, prov = _assign_with_prov(traj, spec, t0, _resolve, arm_match_fn, arm_strategy)
        arms.setdefault(arm_name, []).append(tid)
        index_times[tid] = t0
        if prov is not None:  # a treatment match — record its method + a bounded sample
            method = prov["method"]
            arm_method_counts.setdefault(arm_name, {})
            arm_method_counts[arm_name][method] = arm_method_counts[arm_name].get(method, 0) + 1
            if method == "substring":
                n_low_conf += 1
            if len(provenance) < provenance_sample:
                provenance.append({"trajectory_id": tid, **prov})
        elif arm_name not in treatment_names and arm_name != (next((a.name for a in spec.arms if a.is_control), None)):
            n_unassigned += 1

    control_names = {a.name for a in spec.arms if a.is_control} or {"control"}
    arm_objs = [
        ArmAssignment(name=name, is_control=(name in control_names), trajectory_ids=sorted(ids))
        for name, ids in sorted(arms.items())
    ]
    n_total = sum(len(ids) for ids in arms.values())
    diagnostics = CohortDiagnostics(
        n_screened=len(by_id), n_eligible=n_eligible,
        n_excluded_immortal=n_excluded_immortal, n_enrolled=n_total,
        anchor=spec.time_zero.anchor, grace_window_hours=spec.time_zero.grace_window_hours,
        landmark_hours=landmark, arm_sizes={a.name: len(a.trajectory_ids) for a in arm_objs},
        leakage_warnings=_leakage_warnings(spec),
        n_skipped_unmeasurable=len(skipped_labels), skipped_eligibility=skipped_labels,
    )
    return CohortResult(
        nct_id=spec.nct_id, dataset=dataset, arms=arm_objs, index_times=index_times,
        n_total=n_total, diagnostics=diagnostics,
        eligibility_decisions=eligibility_decisions, assignment_provenance=provenance,
        arm_method_counts=arm_method_counts, n_unassigned=n_unassigned, n_low_confidence=n_low_conf,
    )


def build_analysis_frame(
    events: "pd.DataFrame",
    cohort: CohortResult,
    spec: TargetTrialSpec,
    *,
    covariates: list[FeatureSpec] | None = None,
) -> "pd.DataFrame":
    """Analysis-ready WIDE frame: one row per cohort trajectory with group,
    time_zero, covariate features (deterministic view), and one binary column
    per outcome (event within the outcome horizon of t0).
    """
    import pandas as pd

    ids = [tid for arm in cohort.arms for tid in arm.trajectory_ids]
    group = {tid: arm.name for arm in cohort.arms for tid in arm.trajectory_ids}
    index_times = {int(k): v for k, v in cohort.index_times.items()}

    frame = pd.DataFrame({"TRAJECTORY_ID": sorted(ids)})
    frame["group"] = frame["TRAJECTORY_ID"].map(group)
    frame["time_zero"] = frame["TRAJECTORY_ID"].map(index_times)

    if covariates:
        sub = events[events["TRAJECTORY_ID"].isin(ids)]
        wide = materialize_wide(sub, covariates, index_times=index_times)
        frame = frame.merge(wide, on="TRAJECTORY_ID", how="left")

    for outcome in spec.outcomes:
        horizon = outcome.horizon_hours
        col = f"outcome_{outcome.name.replace(' ', '_')}"
        feat = FeatureSpec(
            name=col,
            event_type=outcome.event_type,
            event_name=outcome.concept,
            agg=Aggregation.ANY,
            window_hours=(0.0, horizon) if horizon is not None else None,
        )
        sub = events[events["TRAJECTORY_ID"].isin(ids)]
        wide = materialize_wide(sub, [feat], index_times=index_times)
        frame = frame.merge(wide, on="TRAJECTORY_ID", how="left")
        frame[col] = frame[col].fillna(False).astype(bool)

    cohort.feature_columns = [c for c in frame.columns if c != "TRAJECTORY_ID"]
    return frame
