"""Code-based concept matching (#129, worker1) — the correctness foundation.

jpic: string matching cannot soundly decide who got which drug / who has the
condition. This resolves a concept to a validated CODE SET and matches patients by
CODE, with a confidence TIER + matched code on every hit so grouping is auditable
(feeds tte1's #130 assignment-audit + UI). Both MIMIC and eICU.

Confidence tiers (locked, highest->lowest):
  rxnorm_code  — code is in the concept's RxNorm-ingredient code set (drugs)
  icd_hierarchy— code is in the condition's ICD family (structured icd_code)
  ingredient   — code's catalog NAME matched the concept's ingredient/synonym set
                 (the human-reviewed layer; the MATCH itself is still by code)
  name         — exact concept-name match
  substring    — last-resort loose substring (LOW — surfaced AMBER, never buried)

Drugs: concept -> ingredient/synonym set -> the gsn/ndc (MIMIC) / drughiclseqno
(eICU) CODES of the catalog rows carrying that ingredient (brand<->generic roll-up);
patients matched by CODE. Conditions: ICD family (hierarchy prefix + explicit
codes) on the structured icd_code. The resolver is pure; the drug-catalog scan
needs pandas (lazy).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DRUG_CATALOG_CACHE = Path.home() / ".cache" / "tteEngine" / "drug_catalog"

# Confidence tiers (LOCKED) — string values are the single source of truth, shared
# with tte1's canonical contracts.audit.Confidence (#139): the matcher emits these
# strings, the audit schema validates them (no duplicate enum -> no seam drift).
# 'icd_hierarchy' is the diagnosis analogue of 'rxnorm_code' (a structured-code
# match) — coordinated to be added to the locked enum.
RXNORM = "rxnorm_code"
ICD_HIERARCHY = "icd_hierarchy"
INGREDIENT = "ingredient"
NAME = "name"
SUBSTRING = "substring"        # LOW — last resort, rendered amber by the UI
LOW_CONFIDENCE = SUBSTRING

#: tier strength — pick the best match + the UI's amber threshold.
TIER_RANK = {RXNORM: 4, ICD_HIERARCHY: 4, INGREDIENT: 3, NAME: 2, SUBSTRING: 1}


@dataclass(frozen=True)
class CodeMatch:
    """One code that matched a concept + HOW (the per-patient provenance primitive;
    maps onto contracts.audit.MatchProvenance: matched_code=code,
    matched_event_name=name, concept, method)."""
    code: str
    name: str
    method: str          # a locked Confidence value
    concept: str

    @property
    def low_confidence(self) -> bool:
        return self.method == LOW_CONFIDENCE


# --------------------------------------------------------------------------- #
# Conditions — ICD code sets via hierarchy (structured icd_code, not titles)
# --------------------------------------------------------------------------- #
def _norm_icd(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


@dataclass
class IcdCodeSet:
    concept: str
    prefixes: tuple[str, ...] = ()
    exact: frozenset[str] = frozenset()

    def match(self, code: str) -> CodeMatch | None:
        c = _norm_icd(code)
        if c in self.exact or (self.prefixes and c.startswith(self.prefixes)):
            return CodeMatch(code=str(code), name=c, method=ICD_HIERARCHY,
                             concept=self.concept)
        return None

    def matches_any(self, value: str) -> bool:
        """True if ANY comma/semicolon-separated code token is in the family — for
        eICU's multi-code icd9code field ('995.92, A41.9') as well as single codes."""
        return any(self.match(tok) is not None
                   for tok in str(value).replace(";", ",").split(","))

    def mask(self, series) -> "pd.Series":
        """Vectorized family membership over an icd_code Series (single-code, e.g.
        MIMIC diagnoses_icd) — for the loader's cheap prefix prefilter."""
        norm = series.astype(str).str.replace(r"[^A-Za-z0-9]", "", regex=True).str.upper()
        m = norm.isin(self.exact)
        if self.prefixes:
            m = m | norm.str.startswith(self.prefixes)
        return m


#: condition keyword -> ICD family (ICD-10 prefixes + ICD-9 prefixes/explicit).
#: 'sepsis': A40*/A41* (sepsis), R652* (severe sepsis +/- shock), 038* (ICD-9
#: septicemia), 99591/99592 (ICD-9 sepsis/severe), 78552 (ICD-9 septic shock).
ICD_FAMILIES: dict[str, IcdCodeSet] = {
    "sepsis": IcdCodeSet("sepsis", prefixes=("A40", "A41", "R652", "038"),
                         exact=frozenset({"99591", "99592", "78552"})),
    "septic shock": IcdCodeSet("septic shock", prefixes=("R6521",),
                               exact=frozenset({"78552"})),
}


def condition_codeset(condition: str) -> IcdCodeSet | None:
    """Map a free-text condition to a curated ICD family (keyword-driven). v1 covers
    sepsis (the priority); extend the table per condition."""
    c = (condition or "").lower()
    if "septic shock" in c:
        return ICD_FAMILIES["septic shock"]
    if "sepsis" in c or "septic" in c or "septicaemia" in c or "septicemia" in c:
        return ICD_FAMILIES["sepsis"]
    return ICD_FAMILIES.get(c)


def build_dx_matcher(concepts) -> dict:
    """#132 cohort/diagnosis matcher {concept: IcdCodeSet} for the cohort-defining
    + dx-eligibility concepts that map to a curated ICD family. Concepts without a
    family are absent -> the adapter falls back to the resolved (#109) codes."""
    out: dict[str, IcdCodeSet] = {}
    for c in concepts:
        if not c:
            continue
        cs = condition_codeset(c)
        if cs is not None:
            out[c] = cs
    return out


# --------------------------------------------------------------------------- #
# Drugs — ingredient/synonym -> code set (gsn/ndc/HICL), match by CODE
# --------------------------------------------------------------------------- #
#: concept -> ingredient + brand/synonym terms (the reviewed layer; brand<->generic
#: roll-up). The CODE SET is derived from the dataset's drug catalog, so it reflects
#: the real formulations present. Extend per trial.
DRUG_INGREDIENTS: dict[str, set[str]] = {
    "hydrocortisone": {"hydrocortisone", "solu-cortef", "cortef", "a-hydrocort",
                       "hydrocortisone sodium succinate"},
    "methylprednisolone": {"methylprednisolone", "solu-medrol", "medrol"},
    "dexamethasone": {"dexamethasone", "decadron"},
    "prednisone": {"prednisone"},
    "prednisolone": {"prednisolone"},
    "fludrocortisone": {"fludrocortisone", "florinef"},
    "corticosteroid": {"hydrocortisone", "solu-cortef", "cortef", "methylprednisolone",
                       "solu-medrol", "medrol", "dexamethasone", "decadron", "prednisone",
                       "prednisolone", "fludrocortisone", "florinef"},
    "norepinephrine": {"norepinephrine", "levophed", "noradrenaline"},
    "epinephrine": {"epinephrine", "adrenaline"},
    "vasopressin": {"vasopressin", "pitressin", "argipressin"},
    "phenylephrine": {"phenylephrine", "neosynephrine", "neo-synephrine"},
    "dopamine": {"dopamine"},
    "dobutamine": {"dobutamine"},
    "vasopressor": {"norepinephrine", "levophed", "epinephrine", "vasopressin",
                    "pitressin", "phenylephrine", "dopamine", "dobutamine"},
}

#: routes a systemic-intervention concept should EXCLUDE (wrong formulation). Real
#: MIMIC data put 'Hydrocortisone ... Foam'/'Suppository' into the steroid set —
#: these are NOT the systemic drug the protocol means.
_EXCLUDE_NAME_TERMS = ("topical", "cream", "ointment", "ophthalmic", "otic", "nasal",
                       "rectal", "enema", "inhal", "lozenge", "patch", "lotion", "gel",
                       "foam", "suppository", "drops", "spray", "mouthwash", "troche", "eye")

#: combination products where the target ingredient is a NON-systemic ADDITIVE
#: (e.g. Bupivacaine-Epinephrine local anesthetic; Ciprofloxacin-Dexamethasone otic;
#: Cyclopentolate-Phenylephrine mydriatic) — exclude so they don't pollute an arm.
_COMBO_PARTNER_TERMS = ("bupivacaine", "lidocaine", "lignocaine", "ropivacaine",
                        "cyclopentolate", "tropicamide", "ciprofloxacin", "neomycin",
                        "polymyxin", "proparacaine", "tetracaine", "benzocaine",
                        "articaine", "mepivacaine")


@dataclass
class DrugCodeSet:
    concept: str
    codes: dict[str, CodeMatch] = field(default_factory=dict)   # normalized code -> match

    def match(self, code: str) -> CodeMatch | None:
        return self.codes.get(_norm_code(code))


def _norm_code(code) -> str:
    return re.sub(r"\s+", "", str(code)).upper().lstrip("0") or "0"


def drug_codeset(concept: str, drug_catalog: list[dict], *, code_fields=("gsn", "ndc"),
                 exclude_wrong_route: bool = True) -> DrugCodeSet:
    """Resolve a drug concept to its CODE SET from a dataset's drug catalog
    (rows of {name, <code_fields>}). A catalog row joins the set if its name carries
    one of the concept's ingredient/synonym terms; its codes are added with method
    'ingredient'. The set is built once (reviewable); patients then match by CODE.

    `code_fields`: ('gsn','ndc') for MIMIC, ('drughiclseqno',) for eICU."""
    ingredients = DRUG_INGREDIENTS.get(concept.lower(), {concept.lower()})
    out = DrugCodeSet(concept=concept)
    for row in drug_catalog:
        name = str(row.get("name", "")).lower()
        if not name:
            continue
        if exclude_wrong_route and (any(x in name for x in _EXCLUDE_NAME_TERMS)
                                    or any(p in name for p in _COMBO_PARTNER_TERMS)):
            continue  # wrong route / combination product -> not the systemic drug
        if not any(ing in name for ing in ingredients):
            continue
        for fld in code_fields:
            code = row.get(fld)
            if code is None or str(code).strip() in ("", "nan"):
                continue
            out.codes[_norm_code(code)] = CodeMatch(
                code=str(code), name=row.get("name", ""),
                method=INGREDIENT, concept=concept)
    return out


def build_drug_catalog(dataset: str, *, root: str | None = None, chunksize: int = 1_000_000,
                       cache_dir: str | Path | None = None, refresh: bool = False) -> list[dict]:
    """Scan the dataset's drug table ONCE for the distinct (name, codes) catalog used
    to build drug code sets: MIMIC prescriptions (drug, gsn, ndc); eICU medication
    (drugname, drughiclseqno). Cached as JSON under `cache_dir` ($HOME by default
    when caching) so the ~minute scan happens once. pandas (lazy)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cpath = cache_dir / f"drug_catalog_{dataset}.json"
        if cpath.exists() and not refresh:
            return json.loads(cpath.read_text())

    import pandas as pd

    from tteEngine.adapters.live_loader import EICU_ROOT, MIMIC_ROOT

    if dataset == "MIMIC-IV":
        path = f"{root or MIMIC_ROOT}/hosp/prescriptions.csv.gz"
        cols, name_col = ["drug", "gsn", "ndc"], "drug"
    elif dataset == "eICU-CRD":
        path = f"{root or EICU_ROOT}/medication.csv.gz"
        cols, name_col = ["drugname", "drughiclseqno"], "drugname"
    else:
        raise ValueError(f"no drug catalog for dataset {dataset!r}")

    seen: dict[tuple, dict] = {}
    for chunk in pd.read_csv(path, usecols=cols, dtype=str, chunksize=chunksize):
        for row in chunk.itertuples(index=False):
            d = row._asdict()
            key = tuple(d.get(c) for c in cols)
            if key not in seen:
                seen[key] = {"name": d.get(name_col), **{c: d.get(c) for c in cols if c != name_col}}
    catalog = list(seen.values())
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cpath.write_text(json.dumps(catalog))
    return catalog


#: drug code fields per dataset (MIMIC gsn/ndc; eICU HICL).
DRUG_CODE_FIELDS = {"MIMIC-IV": ("gsn", "ndc"), "eICU-CRD": ("drughiclseqno",)}


def build_drug_matcher(spec, dataset: str, *, catalog: list[dict] | None = None,
                       root: str | None = None, cache_dir: str | Path | None = None) -> dict:
    """Build the per-arm-intervention drug matcher {concept -> DrugCodeSet} for a
    trial in a dataset, from the (cached) drug catalog. SEAM-INDEPENDENT (#131): this
    is the matcher; the adapter med-EMIT step consumes it once the seam is locked."""
    cat = catalog if catalog is not None else build_drug_catalog(dataset, root=root, cache_dir=cache_dir)
    fields = DRUG_CODE_FIELDS.get(dataset, ("gsn", "ndc"))
    concepts = {iv for arm in spec.arms for iv in arm.intervention_concepts if iv}
    return {c: drug_codeset(c, cat, code_fields=fields) for c in concepts}


# --------------------------------------------------------------------------- #
# Emit into the canonical audit schema (contracts.audit, #140/#144) — no 2nd schema.
# (Re-added: this raced the #141 merge and was stranded; #144 supplied the fields.)
# --------------------------------------------------------------------------- #
#: (category, dataset) -> source table the match came from (for provenance/UI).
SOURCE_TABLES = {
    ("drug", "MIMIC-IV"): "prescriptions", ("drug", "eICU-CRD"): "medication",
    ("diagnosis", "MIMIC-IV"): "diagnoses_icd", ("diagnosis", "eICU-CRD"): "diagnosis",
    ("lab", "MIMIC-IV"): "labevents", ("lab", "eICU-CRD"): "lab",
}


def to_match_provenance(match: "CodeMatch", *, trajectory_id: int, arm: str,
                        t_rel_hours: float | None = None, source_table: str | None = None):
    """Build a contracts.audit.MatchProvenance (#135/#140) from a matcher CodeMatch +
    cohort context. The matcher supplies code/name/concept/method; the cohort seam
    supplies trajectory/arm/time. One canonical schema — this is the emit point."""
    from tteEngine.contracts.audit import Confidence, MatchProvenance

    return MatchProvenance(
        trajectory_id=int(trajectory_id), arm=arm,
        matched_event_name=match.name, matched_code=match.code, concept=match.concept,
        method=Confidence(match.method), t_rel_hours=t_rel_hours, source_table=source_table)


def med_event_value(*, raw_name, code, method, dose=None, source_table=None) -> str:
    """The EVENT_VALUE JSON the adapter writes for a code-matched medication event
    (seam A): {raw_name, code, method, dose, source_table}. The cohort seam reads it
    back via provenance_from_event_value."""
    return json.dumps({"raw_name": None if raw_name is None else str(raw_name),
                       "code": None if code is None else str(code), "method": method,
                       "dose": None if dose is None else str(dose), "source_table": source_table})


def provenance_from_event_value(event_value: str, *, trajectory_id: int, arm: str,
                                concept: str | None = None, t_rel_hours: float | None = None):
    """Build a contracts.audit.MatchProvenance from a code-matched med event's
    EVENT_VALUE JSON (seam A) + cohort context — the helper tte1's arm_match_fn uses."""
    from tteEngine.contracts.audit import Confidence, MatchProvenance

    try:
        meta = json.loads(event_value) if event_value else {}
    except (json.JSONDecodeError, TypeError):
        meta = {}
    method = meta.get("method") or SUBSTRING
    return MatchProvenance(
        trajectory_id=int(trajectory_id), arm=arm,
        matched_event_name=meta.get("raw_name"), matched_code=meta.get("code"),
        concept=concept, method=Confidence(method) if method in {c for c in
                                                                  (RXNORM, ICD_HIERARCHY, INGREDIENT, NAME, SUBSTRING)} else Confidence(SUBSTRING),
        t_rel_hours=t_rel_hours, source_table=meta.get("source_table"))


def assign_med_concepts(df, code_fields, drug_matcher: dict):
    """Vectorized: which arm concept (+ matched code/method) each medication row
    code-matches, via the drug_matcher {concept: DrugCodeSet}. First concept whose
    code set contains any of the row's code-field values wins. Returns three Series
    (concept, matched_code, method) aligned to df; None where no code match."""
    import pandas as pd

    concept = pd.Series([None] * len(df), index=df.index, dtype=object)
    matched_code = pd.Series([None] * len(df), index=df.index, dtype=object)
    method = pd.Series([None] * len(df), index=df.index, dtype=object)
    norm = {fld: df[fld].map(_norm_code) for fld in code_fields if fld in df.columns}
    for c, cs in drug_matcher.items():
        codeset = cs.codes
        if not codeset:
            continue
        for ncol in norm.values():
            hit = ncol.isin(codeset) & concept.isna()
            if hit.any():
                concept.loc[hit] = c
                matched_code.loc[hit] = ncol[hit].map(lambda k: codeset[k].code)
                method.loc[hit] = ncol[hit].map(lambda k: codeset[k].method)
    return concept, matched_code, method


__all__ = [
    "RXNORM", "ICD_HIERARCHY", "INGREDIENT", "NAME", "SUBSTRING",
    "TIER_RANK", "LOW_CONFIDENCE", "CodeMatch",
    "IcdCodeSet", "ICD_FAMILIES", "condition_codeset", "build_dx_matcher",
    "DrugCodeSet", "DRUG_INGREDIENTS", "drug_codeset", "build_drug_catalog",
    "DRUG_CODE_FIELDS", "build_drug_matcher", "DRUG_CATALOG_CACHE",
    "SOURCE_TABLES", "to_match_provenance", "med_event_value",
    "provenance_from_event_value", "assign_med_concepts",
]
