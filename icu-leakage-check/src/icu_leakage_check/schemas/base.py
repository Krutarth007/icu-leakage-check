"""Structures describing how each source table behaves in time.

A :class:`TableSpec` records which columns identify the patient and encounter,
which column carries the clinical time of an event, which carries the time it
became available to clinicians, which columns are only known in hindsight and
which directly encode an outcome. Checks use these specifications to trace a
feature back to its source and decide whether it could have been known at the
moment of prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Timing classes
TIMED = "timed"            # every row has an event time
DATE_ONLY = "date_only"    # rows only carry a calendar date
UNTIMED = "untimed"        # rows have no time; usually finalised at discharge
STATIC = "static"          # one row per patient/encounter/stay with descriptive fields
FIRST_DAY = "first_day"    # summaries of the first 24 hours of the stay
LINKED = "linked"          # detail rows that inherit their time from a parent table
DICTIONARY = "dictionary"  # lookup tables with no patient data

# Column roles reported by DatasetSchema.column_roles
ROLE_ID = "id"
ROLE_EVENT_TIME = "event_time"
ROLE_AVAILABLE_TIME = "available_time"
ROLE_END_TIME = "end_time"
ROLE_HINDSIGHT = "hindsight"
ROLE_OUTCOME = "outcome"
ROLE_FIRST_DAY = "first_day"
ROLE_CALENDAR = "calendar"
ROLE_VALUE = "value"
ROLE_OTHER = "other"


@dataclass(frozen=True)
class TableSpec:
    name: str
    description: str
    timing: str
    module: Optional[str] = None
    id_columns: Tuple[str, ...] = ()
    entity_column: Optional[str] = None
    event_time: Optional[str] = None
    event_time_fallback: Optional[str] = None
    available_time: Optional[str] = None
    available_time_fallback: Optional[str] = None
    end_time: Optional[str] = None
    variable_column: Optional[str] = None
    value_column: Optional[str] = None
    wide_value_columns: Tuple[str, ...] = ()
    datetime_columns: Tuple[str, ...] = ()
    hindsight_columns: Tuple[str, ...] = ()
    outcome_columns: Tuple[str, ...] = ()
    first_day_columns: Tuple[str, ...] = ()
    calendar_columns: Tuple[str, ...] = ()
    proxy_text_columns: Tuple[str, ...] = ()
    code_columns: Tuple[str, ...] = ()
    whole_table_is_proxy: bool = False
    notes: str = ""

    @property
    def key(self) -> str:
        return self.name.lower()

    def time_columns(self) -> List[str]:
        cols = [
            self.event_time,
            self.event_time_fallback,
            self.available_time,
            self.available_time_fallback,
            self.end_time,
        ]
        return [c for c in cols if c]

    def role_of(self, column: str) -> str:
        c = column.lower()
        if c in self.outcome_columns:
            return ROLE_OUTCOME
        if c in self.hindsight_columns:
            return ROLE_HINDSIGHT
        if c in self.first_day_columns:
            return ROLE_FIRST_DAY
        if c in self.calendar_columns:
            return ROLE_CALENDAR
        if c in self.id_columns:
            return ROLE_ID
        if c in (self.event_time, self.event_time_fallback):
            return ROLE_EVENT_TIME
        if c in (self.available_time, self.available_time_fallback):
            return ROLE_AVAILABLE_TIME
        if c == self.end_time:
            return ROLE_END_TIME
        if c == self.value_column or c in self.wide_value_columns:
            return ROLE_VALUE
        return ROLE_OTHER


@dataclass
class DatasetSchema:
    name: str
    time_kind: str  # "datetime" or "minutes"
    patient_id: str
    encounter_id: str
    stay_id: str
    site_id: Optional[str]
    tables: Dict[str, TableSpec] = field(default_factory=dict)
    first_day_minutes: float = 1440.0
    notes: str = ""

    def add(self, spec: TableSpec) -> None:
        self.tables[spec.key] = spec

    def table(self, name: str) -> TableSpec:
        key = str(name).lower()
        for suffix in (".csv.gz", ".csv", ".parquet"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
        if key not in self.tables:
            raise KeyError(f"{self.name} has no table named {name!r}")
        return self.tables[key]

    def has_table(self, name: str) -> bool:
        try:
            self.table(name)
            return True
        except KeyError:
            return False

    def table_names(self, include_dictionaries: bool = True) -> List[str]:
        return [
            t.name
            for t in self.tables.values()
            if include_dictionaries or t.timing != DICTIONARY
        ]

    def column_roles(self, column: str) -> List[Tuple[str, str]]:
        """Every (table, role) pair in which a column name appears with a known role."""
        c = column.lower()
        out = []
        for t in self.tables.values():
            role = t.role_of(c)
            if role != ROLE_OTHER:
                out.append((t.name, role))
        return out

    def hindsight_columns(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for t in self.tables.values():
            for c in t.hindsight_columns + t.outcome_columns:
                out.setdefault(c, []).append(t.name)
        return out

    def summary(self):
        import pandas as pd

        rows = []
        for t in self.tables.values():
            rows.append(
                {
                    "table": t.name,
                    "module": t.module,
                    "timing": t.timing,
                    "entity": t.entity_column,
                    "event_time": t.event_time,
                    "available_time": t.available_time,
                    "end_time": t.end_time,
                    "outcome_columns": ", ".join(t.outcome_columns),
                    "hindsight_columns": ", ".join(t.hindsight_columns),
                    "first_day_columns": len(t.first_day_columns),
                    "notes": t.notes,
                }
            )
        return pd.DataFrame(rows)


def parse_source(source: str) -> Tuple[str, Optional[str]]:
    """Split ``'table.column'`` or ``'table'`` into its parts."""
    s = str(source).strip()
    if "." in s:
        table, column = s.split(".", 1)
        return table.strip(), column.strip() or None
    return s, None


def lower_tuple(values: Iterable[str]) -> Tuple[str, ...]:
    return tuple(v.lower() for v in values)
