"""adapters: per-DB extraction to the canonical 5-col stream (#6 MIMIC / #7 eICU /
#8 MGB-gated, owner: worker1).

Stub — built via PR. Each adapter consumes an ExtractionPlan + the vocab layer
and emits a canonical event stream that passes common_format.validate_canonical.
#6 generalizes EHR-DE/MIMIC-IV/extraction_v1.py; #8 is build-now/test-later.
"""
