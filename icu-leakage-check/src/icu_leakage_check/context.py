"""Normalised inputs shared by all checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .schemas import DatasetSchema, get_schema
from .schemas.base import parse_source
from .splits import TEST, TRAIN, VALID, normalise_split_labels

STRATEGIES = ("random", "patient", "temporal", "site", "cross_validation", None)


@dataclass
class AuditContext:
    data: pd.DataFrame
    label: Optional[str] = None
    split: Optional[pd.Series] = None
    patient_id: Optional[str] = None
    encounter_id: Optional[str] = None
    stay_id: Optional[str] = None
    site_id: Optional[str] = None
    index_time: Optional[str] = None
    outcome_time: Optional[str] = None
    eligibility_time: Optional[str] = None
    calendar_time: Optional[str] = None
    schema: Optional[DatasetSchema] = None
    feature_columns: List[str] = field(default_factory=list)
    feature_sources: Dict[str, str] = field(default_factory=dict)
    events: Optional[pd.DataFrame] = None
    event_entity: Optional[str] = None
    raw_events: Optional[pd.DataFrame] = None
    feature_builder: Optional[Callable] = None
    preprocessor: Any = None
    preprocessor_columns: Optional[List[str]] = None
    scores: Optional[np.ndarray] = None
    threshold: Optional[float] = None
    split_strategy: Optional[str] = None
    declarations: Dict[str, Any] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- helpers
    @property
    def has_split(self) -> bool:
        return self.split is not None and self.split.notna().any()

    def split_names(self) -> List[str]:
        if not self.has_split:
            return []
        order = {TRAIN: 0, VALID: 1, TEST: 2}
        names = list(pd.unique(self.split.dropna()))
        return sorted(names, key=lambda n: (order.get(n, 3), str(n)))

    def evaluation_splits(self) -> List[str]:
        return [s for s in self.split_names() if s != TRAIN]

    def rows(self, split_name: str) -> pd.DataFrame:
        return self.data.loc[self.split == split_name]

    @property
    def id_columns(self) -> List[str]:
        return [c for c in (self.patient_id, self.encounter_id, self.stay_id, self.site_id) if c]

    def entity_key(self) -> Optional[str]:
        return self.event_entity or self.stay_id or self.encounter_id or self.patient_id

    def label_values(self) -> Optional[pd.Series]:
        if not self.label:
            return None
        return pd.to_numeric(self.data[self.label], errors="coerce")

    def is_binary_label(self) -> bool:
        y = self.label_values()
        if y is None:
            return False
        vals = set(pd.unique(y.dropna()))
        return len(vals) == 2 and vals <= {0, 1, 0.0, 1.0}

    def index_series(self) -> Optional[pd.Series]:
        if not self.index_time:
            return None
        return self.data[self.index_time]

    def time_is_datetime(self) -> bool:
        s = self.index_series()
        return s is not None and pd.api.types.is_datetime64_any_dtype(s)

    def source_of(self, feature: str):
        src = self.feature_sources.get(feature)
        if src is None:
            # allow prefix mapping such as {"lab_": "lab"}
            for k, v in self.feature_sources.items():
                if k.endswith("*") and feature.startswith(k[:-1]):
                    src = v
                    break
        if src is None:
            return None, None
        return parse_source(src)


def _infer_features(data: pd.DataFrame, exclude: Sequence[str]) -> List[str]:
    """Every column except identifiers, times, label, split and scores; raw datetimes are skipped."""
    ex = {c for c in exclude if c}
    return [
        c for c in data.columns
        if c not in ex and not pd.api.types.is_datetime64_any_dtype(data[c]) and not pd.api.types.is_timedelta64_dtype(data[c])
    ]


def build_context(
    data: pd.DataFrame,
    label: Optional[str] = None,
    split=None,
    patient_id: Optional[str] = None,
    encounter_id: Optional[str] = None,
    stay_id: Optional[str] = None,
    site_id: Optional[str] = None,
    index_time: Optional[str] = None,
    outcome_time: Optional[str] = None,
    eligibility_time: Optional[str] = None,
    calendar_time: Optional[str] = None,
    dataset=None,
    feature_columns: Optional[Sequence[str]] = None,
    feature_sources: Optional[Mapping[str, str]] = None,
    events: Optional[pd.DataFrame] = None,
    event_entity: Optional[str] = None,
    raw_events: Optional[pd.DataFrame] = None,
    feature_builder: Optional[Callable] = None,
    preprocessor: Any = None,
    preprocessor_columns: Optional[Sequence[str]] = None,
    scores=None,
    threshold: Optional[float] = None,
    split_strategy: Optional[str] = None,
    declarations: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    extra_exclude: Sequence[str] = (),
) -> AuditContext:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame with one row per prediction instance")
    df = data.reset_index(drop=True)
    for name, col in (
        ("label", label), ("patient_id", patient_id), ("encounter_id", encounter_id), ("stay_id", stay_id),
        ("site_id", site_id), ("index_time", index_time), ("outcome_time", outcome_time),
        ("eligibility_time", eligibility_time), ("calendar_time", calendar_time),
    ):
        if col is not None and col not in df.columns:
            raise KeyError(f"{name}={col!r} is not a column of data")

    split_col_name = None
    if split is None:
        split_series = None
    elif isinstance(split, str):
        if split not in df.columns:
            raise KeyError(f"split={split!r} is not a column of data")
        split_series = normalise_split_labels(df[split])
        split_col_name = split
    else:
        arr = pd.Series(np.asarray(split), index=df.index)
        if len(arr) != len(df):
            raise ValueError("split array must have one entry per row of data")
        split_series = normalise_split_labels(arr)

    if split_strategy not in STRATEGIES:
        raise ValueError(f"split_strategy must be one of {STRATEGIES}")

    score_col = None
    score_arr = None
    if scores is not None:
        if isinstance(scores, str):
            if scores not in df.columns:
                raise KeyError(f"scores={scores!r} is not a column of data")
            score_col = scores
            score_arr = pd.to_numeric(df[scores], errors="coerce").to_numpy()
        else:
            score_arr = np.asarray(scores, dtype=float)
            if len(score_arr) != len(df):
                raise ValueError("scores must have one value per row of data")

    exclude = [label, split_col_name, patient_id, encounter_id, stay_id, site_id, index_time,
               outcome_time, eligibility_time, calendar_time, score_col, *extra_exclude]
    feats = list(feature_columns) if feature_columns is not None else _infer_features(df, exclude)
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"feature_columns not in data: {missing[:10]}")

    return AuditContext(
        data=df,
        label=label,
        split=split_series,
        patient_id=patient_id,
        encounter_id=encounter_id,
        stay_id=stay_id,
        site_id=site_id,
        index_time=index_time,
        outcome_time=outcome_time,
        eligibility_time=eligibility_time,
        calendar_time=calendar_time,
        schema=get_schema(dataset),
        feature_columns=feats,
        feature_sources=dict(feature_sources or {}),
        events=events,
        event_entity=event_entity,
        raw_events=raw_events,
        feature_builder=feature_builder,
        preprocessor=preprocessor,
        preprocessor_columns=list(preprocessor_columns) if preprocessor_columns is not None else None,
        scores=score_arr,
        threshold=threshold,
        split_strategy=split_strategy,
        declarations=dict(declarations or {}),
        options=dict(options or {}),
    )
