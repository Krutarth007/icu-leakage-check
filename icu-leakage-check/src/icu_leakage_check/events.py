"""Convert source tables into a common long event format.

Every event row carries the columns below. Times are pandas datetimes for
MIMIC-IV and numeric minute offsets for eICU-CRD; a single event frame should
use one representation throughout.

=================  ============================================================
``entity_id``      unit of prediction the event belongs to (stay, admission ...)
``variable``       what was measured or done (item id, lab name, vital name ...)
``value``          recorded value (numeric where possible)
``event_time``     clinical time of the event
``available_time`` earliest time the value was available to clinicians
``end_time``       end of an interval event, otherwise missing
``source``         ``table.column`` the value came from
``hindsight``      True when the source column is only known in hindsight
=================  ============================================================
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .schemas import DatasetSchema, get_schema
from .schemas.base import DATE_ONLY, DICTIONARY, FIRST_DAY, LINKED, STATIC, UNTIMED

EVENT_COLUMNS = [
    "entity_id",
    "variable",
    "value",
    "event_time",
    "available_time",
    "end_time",
    "source",
    "hindsight",
]


def _later(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise later of two time series, ignoring missing values."""
    out = a.copy()
    take_b = b.notna() & (a.isna() | (b > a))
    out[take_b] = b[take_b]
    return out


def _empty_like(time_kind: str) -> pd.Series:
    return pd.Series(pd.NaT) if time_kind == "datetime" else pd.Series(np.nan)


def events_from_table(
    df: pd.DataFrame,
    table: str,
    dataset: Union[str, DatasetSchema],
    entity_column: Optional[str] = None,
    variable_column: Optional[str] = None,
    value_columns: Optional[Sequence[str]] = None,
    encounter_end: Optional[Mapping] = None,
    date_policy: str = "end_of_day",
) -> pd.DataFrame:
    """Turn a raw table into canonical events using the dataset registry.

    Parameters
    ----------
    df:
        Table as returned by :func:`icu_leakage_check.io.read_table`.
    table:
        Table name in the registry (for example ``"labevents"`` or ``"lab"``).
    entity_column:
        Column that identifies the prediction unit. Defaults to the table's
        natural entity (``stay_id``, ``hadm_id`` or ``patientunitstayid``).
        Join a stay identifier onto hospital-level tables first if you predict
        per ICU stay.
    variable_column, value_columns:
        Override the registry defaults. Several ``value_columns`` are melted
        into separate variables.
    encounter_end:
        Mapping from entity to discharge time. Used as the availability time
        of undated sources such as billing codes; without it their
        availability is left missing, which the audit reports as undated.
    date_policy:
        ``"end_of_day"`` (default) treats date-only rows as available at the
        end of that day, the conservative choice. ``"start_of_day"`` uses
        midnight and will usually leak same-day information.
    """
    schema = get_schema(dataset)
    if schema is None:
        raise ValueError("events_from_table needs a dataset registry ('mimic-iv' or 'eicu')")
    spec = schema.table(table)
    if spec.timing == DICTIONARY:
        raise ValueError(f"{spec.name} is a dictionary table and has no patient events")
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]
    ent = (entity_column or spec.entity_column or "").lower()
    if ent not in data.columns:
        raise KeyError(f"Entity column {ent!r} not in table {spec.name}")

    # Values to melt
    if value_columns is not None:
        vals = [c.lower() for c in value_columns]
    elif spec.wide_value_columns:
        vals = [c for c in spec.wide_value_columns if c in data.columns]
    elif spec.timing in (FIRST_DAY, STATIC):
        skip = set(spec.id_columns) | set(spec.time_columns()) | set(spec.calendar_columns)
        vals = [c for c in data.columns if c not in skip and c != ent]
    elif spec.value_column and spec.value_column in data.columns:
        vals = [spec.value_column]
    else:
        vals = []
    var_col = (variable_column or spec.variable_column)
    var_col = var_col.lower() if var_col else None
    if var_col and var_col not in data.columns:
        var_col = None

    # Time columns
    tk = schema.time_kind

    def col_or_nan(name):
        if name and name in data.columns:
            return data[name]
        return pd.Series(pd.NaT if tk == "datetime" else np.nan, index=data.index)

    if spec.timing == FIRST_DAY:
        start = pd.Series(0.0 if tk == "minutes" else pd.NaT, index=data.index)
        avail = pd.Series(float(schema.first_day_minutes) if tk == "minutes" else pd.NaT, index=data.index)
        event_time, available_time, end_time = start, avail, avail.copy()
    elif spec.timing in (UNTIMED, LINKED):
        event_time = col_or_nan(None)
        if encounter_end is not None:
            available_time = data[ent].map(encounter_end)
        else:
            available_time = col_or_nan(None)
        end_time = col_or_nan(None)
    else:
        event_time = col_or_nan(spec.event_time)
        if spec.event_time_fallback:
            event_time = event_time.fillna(col_or_nan(spec.event_time_fallback))
        available_time = col_or_nan(spec.available_time)
        if spec.available_time_fallback:
            fb = col_or_nan(spec.available_time_fallback)
            if tk == "datetime" and spec.timing != DATE_ONLY and spec.available_time_fallback.endswith("date"):
                fb = pd.to_datetime(fb, errors="coerce") + pd.Timedelta(days=1)
            available_time = available_time.fillna(fb)
        end_time = col_or_nan(spec.end_time)
        if tk == "datetime":
            event_time = pd.to_datetime(event_time, errors="coerce")
            available_time = pd.to_datetime(available_time, errors="coerce")
            end_time = pd.to_datetime(end_time, errors="coerce")
        if spec.timing == DATE_ONLY and tk == "datetime":
            event_time = event_time.dt.normalize()
            if date_policy == "end_of_day":
                available_time = event_time + pd.Timedelta(days=1)
            elif date_policy == "start_of_day":
                available_time = event_time
            else:
                raise ValueError("date_policy must be 'end_of_day' or 'start_of_day'")
        available_time = _later(event_time, available_time) if available_time.notna().any() else event_time.copy()

    hindsight_cols = set(spec.hindsight_columns) | set(spec.outcome_columns)
    frames = []
    base = pd.DataFrame({
        "entity_id": data[ent].values,
        "event_time": event_time.values,
        "available_time": available_time.values,
        "end_time": end_time.values,
    })
    if not vals:
        variable = data[var_col].astype(str).values if var_col else np.repeat(spec.name, len(data))
        f = base.copy()
        f["variable"] = variable
        f["value"] = 1.0
        f["source"] = spec.name
        f["hindsight"] = spec.timing == UNTIMED
        frames.append(f)
    else:
        for v in vals:
            if v not in data.columns:
                continue
            f = base.copy()
            raw = data[v]
            num = pd.to_numeric(raw, errors="coerce")
            f["value"] = num.where(num.notna() | raw.isna(), raw).values
            if var_col and len(vals) == 1 and spec.timing not in (FIRST_DAY, STATIC):
                f["variable"] = data[var_col].astype(str).values
            elif var_col and len(vals) > 1 and spec.timing not in (FIRST_DAY, STATIC):
                f["variable"] = (data[var_col].astype(str) + "__" + v).values
            else:
                f["variable"] = v
            f["source"] = f"{spec.name}.{v}"
            f["hindsight"] = (v in hindsight_cols) or spec.timing == UNTIMED
            if spec.timing == STATIC and v in hindsight_cols:
                if encounter_end is not None:
                    f["available_time"] = data[ent].map(encounter_end).values
                else:
                    f["available_time"] = (
                        pd.Series([pd.NaT] * len(f)).values if tk == "datetime" else np.full(len(f), np.nan)
                    )
            frames.append(f)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)
    out = out.dropna(subset=["value"]) if len(out) else out
    return out[EVENT_COLUMNS].reset_index(drop=True)


def index_times_from_start(
    stays: pd.DataFrame,
    entity_column: str,
    start_column: Optional[str] = None,
    hours: float = 24.0,
) -> pd.DataFrame:
    """Prediction times a fixed number of hours after a start time.

    For datetime data pass ``start_column`` (for example ``intime``). For
    minute offsets relative to unit admission (eICU-CRD) leave
    ``start_column`` as ``None`` and the index time is ``hours * 60``.
    """
    ent = entity_column.lower()
    s = stays.copy()
    s.columns = [c.lower() for c in s.columns]
    if start_column is None:
        t = pd.Series(float(hours) * 60.0, index=s.index)
    else:
        t = pd.to_datetime(s[start_column.lower()], errors="coerce") + pd.Timedelta(hours=hours)
    return pd.DataFrame({"entity_id": s[ent].values, "index_time": t.values}).drop_duplicates("entity_id")


def attach_entity(
    events_df: pd.DataFrame,
    stays: pd.DataFrame,
    from_column: str,
    to_column: str,
    start_column: str,
    end_column: str,
) -> pd.DataFrame:
    """Re-key hospital-level events to ICU stays by time overlap.

    Useful for MIMIC-IV laboratory events, which carry ``hadm_id`` but not
    ``stay_id``: each event is assigned to the stay of the same admission whose
    ``[start, end]`` interval contains its event time.
    """
    s = stays[[from_column, to_column, start_column, end_column]].copy()
    s.columns = ["entity_id", "_new", "_start", "_end"]
    m = events_df.merge(s, on="entity_id", how="inner")
    keep = (m["event_time"] >= m["_start"]) & (m["event_time"] <= m["_end"])
    m = m.loc[keep].copy()
    m["entity_id"] = m["_new"]
    return m.drop(columns=["_new", "_start", "_end"]).reset_index(drop=True)
