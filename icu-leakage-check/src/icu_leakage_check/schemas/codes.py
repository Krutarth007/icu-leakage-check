"""Codes, text patterns and name patterns that indicate outcome information.

These lists are deliberately conservative. A match is a prompt to trace the
feature to its source, not proof of leakage on its own; the checks combine
matches with timing information when it is available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ProxyCode:
    code: str          # normalised: upper case, no dot
    system: str        # "ICD-10-CM" or "ICD-9-CM"
    meaning: str
    prefix: bool       # True when every code starting with ``code`` matches
    strength: str      # "strong" when the code is almost always recorded near the outcome


PROXY_CODES: Tuple[ProxyCode, ...] = (
    ProxyCode("Z66", "ICD-10-CM", "Do not resuscitate status", False, "strong"),
    ProxyCode("Z515", "ICD-10-CM", "Encounter for palliative care", False, "strong"),
    ProxyCode("G9382", "ICD-10-CM", "Brain death", False, "strong"),
    ProxyCode("R99", "ICD-10-CM", "Ill-defined and unknown cause of mortality", False, "strong"),
    ProxyCode("I46", "ICD-10-CM", "Cardiac arrest", True, "context"),
    ProxyCode("V667", "ICD-9-CM", "Encounter for palliative care", False, "strong"),
    ProxyCode("V4986", "ICD-9-CM", "Do not resuscitate status", False, "strong"),
    ProxyCode("34882", "ICD-9-CM", "Brain death", False, "strong"),
    ProxyCode("4275", "ICD-9-CM", "Cardiac arrest", False, "context"),
    ProxyCode("798", "ICD-9-CM", "Sudden death, cause unknown", True, "strong"),
)

# Free-text patterns for care limitation, end of life and related documentation.
PROXY_TEXT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"comfort\s*(care|measures?)", "comfort care"),
    (r"palliat", "palliative care"),
    (r"hospice", "hospice"),
    (r"\bdnr\b|do[\s\-]*not[\s\-]*resuscitat", "do not resuscitate"),
    (r"\bdni\b|do[\s\-]*not[\s\-]*intubat", "do not intubate"),
    (r"\bno\s*cpr\b", "no CPR"),
    (r"withdraw\w*\s*(of\s*)?(care|support|treatment|therapy|life)", "withdrawal of treatment"),
    (r"withh[eo]ld\w*\s*(of\s*)?(care|support|treatment|therapy)", "withholding of treatment"),
    (r"care\s*limitation|limitation\s*of\s*(care|treatment)|limit(ed)?\s*code", "care limitation"),
    (r"end[\s\-]*of[\s\-]*life", "end of life"),
    (r"brain\s*death|brain\s*dead", "brain death"),
    (r"organ\s*(donat|procure)", "organ donation"),
    (r"goals?\s*of\s*care", "goals of care"),
    (r"code\s*status", "code status"),
    (r"\bcmo\b", "comfort measures only"),
    (r"terminal\s*(wean|extubat)", "terminal weaning"),
    (r"post[\s\-]*mortem|autopsy|expired|deceased", "death documentation"),
)

# Name patterns that suggest a column encodes the outcome itself.
OUTCOME_NAME_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"(^|_)(died|death|deceased|dead|expire[ds]?|expire_flag|mortality|dod|deathtime)(_|$)", "death or mortality field"),
    (r"(^|_)discharge_?status(_|$)|(^|_)dischargestatus(_|$)", "discharge status"),
    (r"(^|_)(actual|observed)_?(icu|hospital|hosp)?_?mortality", "observed mortality"),
)

# Name patterns that suggest a hindsight variable.
HINDSIGHT_NAME_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"(^|_)(los|length_?of_?stay)(_|$)|(icu|hosp|unit|hospital)_?los", "length of stay"),
    (r"disch|discharge", "discharge information"),
    (r"(^|_)outtime(_|$)", "unit or ward exit time"),
    (r"unabridged", "unabridged stay summary"),
    (r"(^|_)actual", "observed outcome"),
    (r"(^|_)drg(_|$)|drg_(severity|mortality|code)", "billing DRG"),
    (r"active_?upon_?discharge", "status at discharge"),
    (r"(whole|entire|total|full)_?(stay|admission|encounter)", "whole-stay summary"),
    (r"(^|_)(stay|admission)_(max|min|mean|sum|total|last|count)(_|$)", "whole-stay summary"),
    (r"(^|_)last_?careunit(_|$)", "last care unit"),
    (r"(^|_)vent(ilation)?_?days(_|$)", "ventilation days"),
)

CODE_HINT_TOKENS = {"ICD", "ICD9", "ICD10", "DX", "DIAG", "DIAGNOSIS", "CODE", "CODES", "PROC", "CCS", "PHECODE", "DRG"}

_COMPILED_TEXT = [(re.compile(p, re.IGNORECASE), label) for p, label in PROXY_TEXT_PATTERNS]
_COMPILED_OUTCOME = [(re.compile(p, re.IGNORECASE), label) for p, label in OUTCOME_NAME_PATTERNS]
_COMPILED_HINDSIGHT = [(re.compile(p, re.IGNORECASE), label) for p, label in HINDSIGHT_NAME_PATTERNS]


def normalise_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def match_proxy_code(code: str) -> Optional[ProxyCode]:
    """Return the proxy definition that a single diagnosis code matches, if any."""
    c = normalise_code(code)
    if not c:
        return None
    for p in PROXY_CODES:
        if (p.prefix and c.startswith(p.code)) or (not p.prefix and c == p.code):
            return p
    return None


def flag_proxy_codes(codes: Iterable[str]) -> pd.DataFrame:
    """Annotate a sequence of diagnosis codes with outcome-proxy meanings.

    Values containing several codes separated by commas, semicolons or spaces
    (as in some multi-centre databases) are split before matching.
    """
    rows = []
    for raw in codes:
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        for part in re.split(r"[,;\s]+", str(raw)):
            m = match_proxy_code(part)
            if m is not None:
                rows.append({"raw": raw, "code": part, "meaning": m.meaning, "system": m.system, "strength": m.strength})
    return pd.DataFrame(rows, columns=["raw", "code", "meaning", "system", "strength"])


def match_proxy_text(text: str) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    for rx, label in _COMPILED_TEXT:
        if rx.search(s):
            return label
    return None


def _name_variants(name: str) -> List[str]:
    s = str(name)
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s).lower()
    snake = re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
    return [snake, s.lower()]


def match_outcome_name(name: str) -> Optional[str]:
    for v in _name_variants(name):
        for rx, label in _COMPILED_OUTCOME:
            if rx.search(v):
                return label
    return None


def match_hindsight_name(name: str) -> Optional[str]:
    for v in _name_variants(name):
        for rx, label in _COMPILED_HINDSIGHT:
            if rx.search(v):
                return label
    return None


def match_code_in_name(name: str) -> Optional[ProxyCode]:
    """Detect proxy codes embedded in one-hot feature names such as ``icd_Z66``.

    Only names that also contain a coding hint token are considered, to avoid
    matching unrelated numbers.
    """
    upper = str(name).upper().replace(".", "")
    tokens = [t for t in re.split(r"[^A-Z0-9]+", upper) if t]
    if not any(t in CODE_HINT_TOKENS or t.startswith(("ICD", "DX")) for t in tokens):
        return None
    for t in tokens:
        cand = t
        for prefix in ("ICD10", "ICD9", "ICD", "DX"):
            if cand.startswith(prefix) and len(cand) > len(prefix):
                cand = cand[len(prefix):]
                break
        m = match_proxy_code(cand)
        if m is not None:
            return m
    return None


def match_proxy_feature_name(name: str) -> Optional[str]:
    """Return a label when a feature name itself refers to an outcome proxy."""
    pc = match_code_in_name(name)
    if pc is not None:
        return f"{pc.meaning} ({pc.system} {pc.code})"
    for v in _name_variants(name):
        spaced = v.replace("_", " ")
        lab = match_proxy_text(spaced)
        if lab is not None:
            return lab
    return None


def find_proxy_items(dictionary: pd.DataFrame, label_column: str = "label", id_column: str = "itemid") -> pd.DataFrame:
    """Search an item dictionary (for example MIMIC-IV d_items) for proxy items.

    Returns the matching rows with an added ``proxy_meaning`` column. Use the
    returned identifiers to exclude or time-restrict those items.
    """
    if label_column not in dictionary.columns:
        raise KeyError(f"{label_column!r} not in dictionary columns")
    meanings = dictionary[label_column].map(match_proxy_text)
    out = dictionary.loc[meanings.notna()].copy()
    out["proxy_meaning"] = meanings[meanings.notna()]
    cols = [c for c in (id_column, label_column, "proxy_meaning") if c in out.columns]
    rest = [c for c in out.columns if c not in cols]
    return out[cols + rest].reset_index(drop=True)


def proxy_text_summary(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, int]:
    """Count rows whose text columns mention an outcome proxy, by meaning."""
    counts: Dict[str, int] = {}
    for col in columns:
        if col not in df.columns:
            continue
        labels = df[col].dropna().astype(str).map(match_proxy_text).dropna()
        for lab, n in labels.value_counts().items():
            counts[lab] = counts.get(lab, 0) + int(n)
    return counts
