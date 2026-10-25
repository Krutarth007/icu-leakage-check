"""Schema registries for supported databases."""

from __future__ import annotations

from typing import Optional, Union

from .base import (
    DATE_ONLY,
    DICTIONARY,
    FIRST_DAY,
    LINKED,
    STATIC,
    TIMED,
    UNTIMED,
    DatasetSchema,
    TableSpec,
    parse_source,
)
from .codes import (
    PROXY_CODES,
    find_proxy_items,
    flag_proxy_codes,
    match_proxy_code,
    match_proxy_feature_name,
    match_proxy_text,
)
from .eicu import EICU
from .mimic_iv import MIMIC_IV

_ALIASES = {
    "mimic-iv": MIMIC_IV,
    "mimic_iv": MIMIC_IV,
    "mimiciv": MIMIC_IV,
    "mimic4": MIMIC_IV,
    "mimic-4": MIMIC_IV,
    "mimic": MIMIC_IV,
    "eicu": EICU,
    "eicu-crd": EICU,
    "eicu_crd": EICU,
}


def get_schema(dataset: Union[str, DatasetSchema, None]) -> Optional[DatasetSchema]:
    """Return the registry for a dataset name, or ``None`` for generic data."""
    if dataset is None:
        return None
    if isinstance(dataset, DatasetSchema):
        return dataset
    key = str(dataset).strip().lower()
    if key in ("", "none", "generic"):
        return None
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Supported: 'mimic-iv', 'eicu', or None for generic data."
        )
    return _ALIASES[key]


__all__ = [
    "DATE_ONLY",
    "DICTIONARY",
    "EICU",
    "FIRST_DAY",
    "LINKED",
    "MIMIC_IV",
    "PROXY_CODES",
    "STATIC",
    "TIMED",
    "UNTIMED",
    "DatasetSchema",
    "TableSpec",
    "find_proxy_items",
    "flag_proxy_codes",
    "get_schema",
    "match_proxy_code",
    "match_proxy_feature_name",
    "match_proxy_text",
    "parse_source",
]
