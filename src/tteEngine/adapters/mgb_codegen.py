"""MGB targeted-extraction CODEGEN (#103, worker1).

MGB is human-gated (no live run from here), so instead of fetching, this GENERATES
a standalone, plan-targeted Snowflake SQL script a credentialed human can review +
run on MGB. It is the generalization jpic wants — one TTE ExtractionPlan -> a
custom MGB extraction — extended to the gated dataset (the analogue of #101's
loaders for MIMIC/eICU).

The MGB Snowflake pipeline already emits the canonical 5-col stream
(TRAJECTORY_ID / TIMESTAMP / EVENT_TYPE / EVENT_NAME / EVENT_VALUE), so the
generated SQL MIRRORS adapters.mgb.extract in Snowflake: cohort = trajectories
with a DIAGNOSIS event matching the cohort codes; a landmark t0 = each
trajectory's first event; keep the requested concepts per event-type; clip to the
extraction window relative to t0. Run it, export the 5 columns, then feed the
result straight to adapters.mgb.extract (or use it as-is — it's already canonical).

No live MGB access, no credentials, no network — pure string generation + tests.
The source table + the concept->code resolution are parameterized (use the #109
vocab index's codes_for to fill real MGB codes).
"""
from __future__ import annotations

from typing import Callable

from tteEngine.contracts.events import EventType
from tteEngine.contracts.extraction_plan import ExtractionPlan
from tteEngine.timing import effective_window

Resolver = Callable[[str], set]
DEFAULT_SOURCE_TABLE = "MGB.CANONICAL_EVENTS"  # set to your MGB 5-col events table


def _identity(concept: str) -> set:
    return {concept}


def _in_list(values) -> str:
    """A safe, deterministic SQL IN-list of single-quoted, escaped string literals."""
    return ", ".join(f"'{v}'" for v in sorted({str(v).replace("'", "''") for v in values}))


def generate_mgb_script(plan: ExtractionPlan, *, resolve: Resolver | None = None,
                        source_table: str = DEFAULT_SOURCE_TABLE) -> str:
    """Generate a Snowflake SQL script that extracts `plan`'s cohort + concepts +
    window from the MGB canonical 5-col events table. Mirrors adapters.mgb.extract.

    `resolve` (concept -> source codes; default identity, pass vocab.resolve or a
    #109 vocab-index resolver for real MGB codes) is applied to the cohort filter
    and each requested concept."""
    resolve = resolve or _identity
    lo, hi = effective_window(plan)

    cohort_codes: set = set()
    for c in plan.cohort_filter_concepts:
        cohort_codes |= set(resolve(c))

    by_type: dict[str, set] = {}
    for req in plan.concepts:
        by_type.setdefault(req.event_type.value, set()).update(resolve(req.concept))

    header = (
        f"-- Auto-generated MGB extraction for {plan.nct_id} (tteEngine #103 codegen).\n"
        "-- DO NOT EDIT BY HAND — regenerate from the ExtractionPlan.\n"
        "-- GATED: review + run on MGB only with data-access authorization. Output is\n"
        "-- the canonical 5-col stream; feed it to tteEngine.adapters.mgb.extract or\n"
        "-- use directly (already canonical).\n"
        f"-- Extraction window: [{lo}h, {hi}h] relative to each trajectory's first event.\n"
    )

    if cohort_codes:
        cohort_cte = (
            "cohort AS (\n"
            "    SELECT DISTINCT TRAJECTORY_ID\n"
            f"    FROM {source_table}\n"
            f"    WHERE EVENT_TYPE = '{EventType.DIAGNOSIS.value}'\n"
            f"      AND EVENT_NAME IN ({_in_list(cohort_codes)})\n"
            ")")
    else:
        cohort_cte = (
            "cohort AS (   -- no cohort filter -> every trajectory\n"
            f"    SELECT DISTINCT TRAJECTORY_ID FROM {source_table}\n"
            ")")

    # per-event-type concept filter (omitted -> keep every concept)
    if by_type:
        clauses = [f"(EVENT_TYPE = '{et}' AND EVENT_NAME IN ({_in_list(codes)}))"
                   for et, codes in sorted(by_type.items()) if codes]
        concept_filter = "  AND (\n        " + "\n     OR ".join(clauses) + "\n  )\n"
    else:
        concept_filter = ""

    return (
        f"{header}\n"
        f"WITH {cohort_cte},\n"
        "windowed AS (\n"
        "    SELECT e.TRAJECTORY_ID, e.TIMESTAMP, e.EVENT_TYPE, e.EVENT_NAME, e.EVENT_VALUE,\n"
        "           MIN(e.TIMESTAMP) OVER (PARTITION BY e.TRAJECTORY_ID) AS T0\n"
        f"    FROM {source_table} e\n"
        "    JOIN cohort USING (TRAJECTORY_ID)\n"
        ")\n"
        "SELECT TRAJECTORY_ID, TIMESTAMP, EVENT_TYPE, EVENT_NAME, EVENT_VALUE\n"
        "FROM windowed\n"
        f"WHERE TIMESTAMP BETWEEN DATEADD('hour', {lo}, T0) AND DATEADD('hour', {hi}, T0)\n"
        f"{concept_filter}"
        "ORDER BY TRAJECTORY_ID, TIMESTAMP;\n"
    )


def write_mgb_script(plan: ExtractionPlan, path, *, resolve: Resolver | None = None,
                     source_table: str = DEFAULT_SOURCE_TABLE) -> str:
    """Generate + write the MGB SQL script to `path`. Returns the SQL string."""
    sql = generate_mgb_script(plan, resolve=resolve, source_table=source_table)
    with open(path, "w") as f:
        f.write(sql)
    return sql


__all__ = ["generate_mgb_script", "write_mgb_script", "DEFAULT_SOURCE_TABLE"]
