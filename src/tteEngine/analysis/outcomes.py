"""Outcome selection for emulation (#111 yield fix, probe).

A ctgov trial lists many outcomes, often with a non-mortality endpoint FIRST
(e.g. 'Health Related Quality of Life (EQ-5D-5L)') that isn't measurable in ICU
EHR. The engine previously emulated ``spec.outcomes[0]`` blindly -> the
materialized ``outcome_<name>`` column is absent -> KeyError -> the whole trial
drops. (In the first real MIMIC run this dropped most of the sepsis set.)

`select_measurable_outcome` picks the outcome to emulate from the ones ACTUALLY
materializable in this dataset's analysis frame, preferring a binary mortality
endpoint (what ICU EHR measures well + what most sepsis RCTs report), then any
binary endpoint, then whatever is left. Returns None only if NO outcome is
measurable here (the caller drops with an explicit reason — never silent).

Import-light (no pandas/analysis deps): pure spec + column-name logic.
"""
from __future__ import annotations

#: substrings that mark a mortality / survival endpoint (ICU-measurable, usually reported)
_MORTALITY_KEYS = ("mortalit", "death", "died", "surviv", "fatal", "in-hospital death")


def outcome_column(name: str) -> str:
    """The analysis-frame column the cohort builder materializes for an outcome —
    must match cohort.builder: ``outcome_<name with spaces->underscores>``."""
    return f"outcome_{name.replace(' ', '_')}"


def is_mortality_outcome(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _MORTALITY_KEYS)


def select_measurable_outcome(spec, available_columns):
    """Pick the OutcomeSpec to emulate among those materializable in this dataset.

    `available_columns`: the analysis frame's columns (so we only pick an outcome
    whose ``outcome_<name>`` column exists). Preference order: binary mortality >
    other binary > first measurable. Returns the chosen OutcomeSpec, or None if no
    declared outcome is measurable in this dataset (caller drops, explicitly)."""
    cols = set(available_columns)
    candidates = [o for o in spec.outcomes if outcome_column(o.name) in cols]
    if not candidates:
        return None

    def rank(o):
        # higher = preferred; mortality first, then binary, then has a horizon
        return (
            is_mortality_outcome(o.name),
            getattr(o, "kind", "binary") == "binary",
            o.horizon_hours is not None,
        )

    return max(candidates, key=rank)
