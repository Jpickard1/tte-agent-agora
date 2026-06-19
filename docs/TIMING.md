# Cross-dataset timing harmonization (#31)

The three datasets keep time differently, so a window or a precision means
something slightly different in each — left implicit, estimands are silently
**incomparable**:

| Dataset   | Clock                       | Timestamp form            | Native precision |
|-----------|-----------------------------|---------------------------|------------------|
| MIMIC-IV  | wall-clock (ICU + hospital) | `datetime64[ns, UTC]`     | sub-minute       |
| eICU-CRD  | **offsets** from unit admit | `EPOCH + offset_minutes`  | minute           |
| MGB       | wall-clock (Snowflake)      | `datetime64[ns, UTC]`     | sub-minute (gated)|

`TimingConfig` (`tteEngine.contracts.timing`) makes the timing **explicit and
shared**, so one config drives every adapter identically.

## The contract

```python
from tteEngine.contracts.timing import TimingConfig, ClockReference, TimePrecision

timing = TimingConfig(
    clock=ClockReference.ICU_ADMISSION,     # which admission anchors t0
    extraction_window_hours=(-48.0, 24.0),  # data-pull window around t0
    lookback_hours=48.0,                    # covariate/eligibility lookback before t0
    washout_hours=0.0,                      # no-prior-exposure window before t0
    grace_window_hours=24.0,                # post-t0 grace for treatment assignment
    precision=TimePrecision.MINUTE,         # COMMON grid all datasets are floored to
)
```

All windows are **hours relative to time-zero**. The validator rejects an
extraction window whose lower bound exceeds its upper bound.

## How it drives the adapters

Attach it to the plan; every adapter (`mimic` / `eicu` / `mgb`) consumes it via
two helpers in `tteEngine.timing`:

```python
plan = ExtractionPlan(nct_id=..., concepts=[...], timing=timing)
events = eicu.extract(plan, tables)   # uses timing's window + precision
```

- **`effective_window(plan)`** — the extraction window an adapter applies:
  `plan.timing.extraction_window_hours` when a `TimingConfig` is set, else the
  legacy `plan.window_hours`.
- **`harmonize_timestamps(df, timing)`** — floors the canonical `TIMESTAMP` of
  every emitted stream to `timing.precision`, so MIMIC (sub-minute), eICU (minute
  offsets) and MGB land on the **same grid**.
- **`to_time_zero_rule(timing)`** — bridges the config to the cohort builder's
  `TimeZeroRule` (same clock + grace), so t0 anchoring uses the same contract as
  extraction.
- **`precision_warnings(timing, dataset)`** — surfaces (never silently coerces)
  when a config asks for finer precision than a dataset natively supports (e.g.
  `second` on eICU, which is minute-only).

## Back-compatibility

`ExtractionPlan.timing` defaults to `None`. With no `TimingConfig`, adapters fall
back to `plan.window_hours` and skip harmonization — existing behavior is
unchanged (the full suite passes untouched). Adopt the contract by setting
`plan.timing`; nothing else changes.
