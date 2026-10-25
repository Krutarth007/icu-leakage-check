"""Leakage-safe ways to split clinical data."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def is_text(s: pd.Series) -> bool:
    """True for object, string and categorical columns."""
    return (
        pd.api.types.is_object_dtype(s)
        or pd.api.types.is_string_dtype(s)
        or isinstance(s.dtype, pd.CategoricalDtype)
    )

TRAIN, VALID, TEST = "train", "valid", "test"

_SPLIT_SYNONYMS = {
    "train": TRAIN, "training": TRAIN, "dev": TRAIN, "development": TRAIN, "derivation": TRAIN, "fit": TRAIN,
    "valid": VALID, "val": VALID, "validation": VALID, "tune": VALID, "tuning": VALID, "calibration": VALID,
    "test": TEST, "testing": TEST, "holdout": TEST, "hold-out": TEST, "external": TEST, "eval": TEST,
    "evaluation": TEST, "internal_test": TEST, "external_test": TEST,
}


def normalise_split_labels(values: pd.Series) -> pd.Series:
    """Map common split names to ``train`` / ``valid`` / ``test``.

    Boolean or 0/1 columns are read as ``is_test``. Unknown labels are kept as
    they are so several evaluation sets can be audited together.
    """
    s = pd.Series(values)
    if s.dtype == bool or set(pd.unique(s.dropna())) <= {0, 1, True, False}:
        return s.map({True: TEST, 1: TEST, False: TRAIN, 0: TRAIN})
    return s.astype(str).str.strip().map(lambda v: _SPLIT_SYNONYMS.get(v.lower(), v))


def _group_labels(df: pd.DataFrame, group: str, stratify: Optional[str]) -> pd.DataFrame:
    if stratify is None:
        g = pd.DataFrame({group: pd.unique(df[group])})
        g["_strat"] = 0
        return g
    g = df.groupby(group)[stratify].max().reset_index()
    g.columns = [group, "_strat"]
    return g


def _can_stratify(labels: pd.Series, n_parts: int) -> bool:
    counts = labels.value_counts()
    return len(counts) > 1 and counts.min() >= max(2, n_parts)


def group_split(
    df: pd.DataFrame,
    group: str,
    test_size: float = 0.2,
    valid_size: float = 0.0,
    stratify: Optional[str] = None,
    random_state: int = 42,
) -> pd.Series:
    """Split so that every group (usually a patient) falls in exactly one split.

    Stratification uses each group's maximum label, so a patient with any
    positive stay counts as positive. Returns a Series of ``train`` /
    ``valid`` / ``test`` aligned with ``df``.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if valid_size < 0 or test_size + valid_size >= 1:
        raise ValueError("valid_size must be >= 0 and test_size + valid_size < 1")
    groups = _group_labels(df, group, stratify)
    strat = groups["_strat"] if stratify and _can_stratify(groups["_strat"], 2) else None
    rest, test = train_test_split(groups, test_size=test_size, random_state=random_state, stratify=strat)
    assignment = {g: TEST for g in test[group]}
    if valid_size > 0:
        rel = valid_size / (1 - test_size)
        strat2 = rest["_strat"] if stratify and _can_stratify(rest["_strat"], 2) else None
        train, valid = train_test_split(rest, test_size=rel, random_state=random_state, stratify=strat2)
        assignment.update({g: VALID for g in valid[group]})
        assignment.update({g: TRAIN for g in train[group]})
    else:
        assignment.update({g: TRAIN for g in rest[group]})
    return df[group].map(assignment).rename("split")


def site_split(
    df: pd.DataFrame,
    site: str,
    test_sites: Optional[Iterable] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.Series:
    """Hold out whole sites (for example eICU-CRD hospitals) for external evaluation."""
    sites = pd.unique(df[site].dropna())
    if test_sites is None:
        rng = np.random.default_rng(random_state)
        n_test = max(1, int(round(len(sites) * test_size)))
        test_sites = rng.choice(sites, size=n_test, replace=False)
    test_set = set(test_sites)
    return df[site].map(lambda s: TEST if s in test_set else TRAIN).rename("split")


def calendar_start_year(values: pd.Series) -> pd.Series:
    """Leading four-digit year of values such as MIMIC-IV ``anchor_year_group`` ('2017 - 2019')."""
    def _first_year(v):
        if pd.isna(v):
            return np.nan
        m = re.search(r"(\d{4})", str(v))
        return float(m.group(1)) if m else np.nan

    return pd.Series(values).map(_first_year)


def temporal_split(
    df: pd.DataFrame,
    time_column: str,
    test_from,
    valid_from=None,
) -> pd.Series:
    """Assign rows at or after ``test_from`` to test (and ``valid_from`` to valid).

    For MIMIC-IV use ``anchor_year_group`` (true calendar period); admission
    dates are shifted per patient and cannot order patients in time. String
    year groups are converted with :func:`calendar_start_year`.
    """
    t = df[time_column]
    if is_text(t):
        t = calendar_start_year(t)
        test_from = float(calendar_start_year(pd.Series([test_from])).iloc[0]) if isinstance(test_from, str) else test_from
        if isinstance(valid_from, str):
            valid_from = float(calendar_start_year(pd.Series([valid_from])).iloc[0])
    out = pd.Series(TRAIN, index=df.index, name="split")
    if valid_from is not None:
        out[t >= valid_from] = VALID
    out[t >= test_from] = TEST
    out[t.isna()] = np.nan
    return out


def describe_split(df: pd.DataFrame, split: str, label: Optional[str] = None, groups: Sequence[str] = ()) -> pd.DataFrame:
    """Rows, prevalence and distinct group counts per split."""
    s = normalise_split_labels(df[split])
    rows = []
    for name in pd.unique(s.dropna()):
        part = df.loc[s == name]
        row = {"split": name, "rows": len(part)}
        if label:
            row["prevalence"] = float(pd.to_numeric(part[label], errors="coerce").mean())
        for g in groups:
            row[f"n_{g}"] = int(part[g].nunique())
        rows.append(row)
    return pd.DataFrame(rows)
