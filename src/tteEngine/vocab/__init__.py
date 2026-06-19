"""vocab: concept/unit normalization layer (#5).

Maps raw EVENT_NAME + value/unit -> cross-DB concept ids (ICD/RxNorm/LOINC) and
normalized units, producing the NormalizedEvent sidecar. Keeps the canonical
5-col structure (contracts.events) decoupled from semantics. Seeds: EHR-DE
clinical_constants (sepsis ICD sets, SOFA), data dictionaries, trialsim
concept_map. Stub — built via PR.
"""
