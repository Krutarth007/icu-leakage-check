"""Build features from canonical events with an explicit prediction-time cut-off.

The default settings are the leakage-safe ones: events are filtered on the
time they became available, interval events are prorated at the prediction
time and nothing after the prediction time is used. Setting ``cutoff='none'``
or ``time_basis='event'`` deliberately produces leaky variants, which is
useful for measuring how much a given shortcut inflates performance.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

SUPPORTED_AGGREGATIONS = ("last", "first", "min", "max", "mean", "median", "std", "sum", "count", "time_since_last")


def _as_index_frame(index_times: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if isinstance(index_times, pd.Series):
        return pd.DataFrame({"entity_id": index_times.index, "index_time": index_times.values})
    df = index_times.copy()
    if "entity_id" not in df.columns or "index_time" not in df.columns:
        raise KeyError("index_times needs columns 'entity_id' and 'index_time'")
    return df[["entity_id", "index_time"]].drop_duplicates("entity_id")


def _to_window(window, time_is_datetime: bool):
    if window is None:
        return None
    if time_is_datetime:
        if isinstance(window, (int, float)):
            return pd.Timedelta(hours=float(window))
        return pd.Timedelta(window)
    if isinstance(window, pd.Timedelta):
        return window.total_seconds() / 60.0
    return float(window)


def select_events(
    events: pd.DataFrame,
    index_times: Union[pd.DataFrame, pd.Series],
    window=None,
    time_basis: str = "available",
    cutoff: str = "index",
    interval_policy: str = "prorate",
    gap=None,
) -> pd.DataFrame:
    """Return the events that a model scored at ``index_time`` may use.

    Parameters
    ----------
    window:
        Look-back length. Hours (number) or a Timedelta for datetime data,
        minutes for offset data. ``None`` uses all history.
    time_basis:
        ``"available"`` filters on ``available_time`` (falls back to
        ``event_time``); ``"event"`` filters on clinical time only and lets
        results that were not yet available slip in.
    cutoff:
        ``"index"`` stops at the prediction time; ``"none"`` keeps the whole
        record including events after the prediction time.
    interval_policy:
        What to do with interval events still running at the prediction time:
        ``"prorate"`` scales numeric values by the elapsed fraction,
        ``"keep_start"`` keeps the value as recorded at its start (use for
        rates), ``"exclude"`` drops them, ``"keep"`` keeps full totals.
    gap:
        Optional blind period before the prediction time, same units as
        ``window``; events inside it are not used.
    """
    idx = _as_index_frame(index_times)
    ev = events.merge(idx, on="entity_id", how="inner")
    is_dt = pd.api.types.is_datetime64_any_dtype(ev["event_time"]) or pd.api.types.is_datetime64_any_dtype(ev["index_time"])
    if time_basis == "available":
        basis = ev["available_time"].where(ev["available_time"].notna(), ev["event_time"])
    elif time_basis == "event":
        basis = ev["event_time"].where(ev["event_time"].notna(), ev["available_time"])
    else:
        raise ValueError("time_basis must be 'available' or 'event'")
    ev["_basis"] = basis

    if cutoff == "none":
        keep = pd.Series(True, index=ev.index)
        horizon = ev["index_time"]
    elif cutoff == "index":
        horizon = ev["index_time"]
        g = _to_window(gap, is_dt)
        if g is not None:
            horizon = horizon - g
        keep = ev["_basis"].notna() & (ev["_basis"] <= horizon)
    else:
        raise ValueError("cutoff must be 'index' or 'none'")

    w = _to_window(window, is_dt)
    if w is not None:
        start_ref = ev["event_time"].where(ev["event_time"].notna(), ev["_basis"])
        keep &= start_ref > (ev["index_time"] - w)

    ev = ev.loc[keep].copy()
    if pd.api.types.is_float_dtype(ev["value"]):
        ev["value"] = ev["value"].astype("float64")  # prorated values must fit regardless of the input precision

    if cutoff == "index" and "end_time" in ev.columns and ev["end_time"].notna().any():
        h = horizon.loc[ev.index]
        running = ev["end_time"].notna() & (ev["end_time"] > h)
        if running.any():
            if interval_policy == "exclude":
                ev = ev.loc[~running].copy()
            elif interval_policy == "prorate":
                num = pd.to_numeric(ev.loc[running, "value"], errors="coerce")
                total = ev.loc[running, "end_time"] - ev.loc[running, "event_time"]
                elapsed = h.loc[running] - ev.loc[running, "event_time"]
                if is_dt:
                    total = total.dt.total_seconds()
                    elapsed = elapsed.dt.total_seconds()
                frac = (elapsed / total.replace(0, np.nan)).clip(0, 1).fillna(1.0)
                ev.loc[running, "value"] = (num * frac).where(num.notna(), ev.loc[running, "value"])
                ev.loc[running, "end_time"] = h.loc[running]
            elif interval_policy == "keep_start":
                ev.loc[running, "end_time"] = h.loc[running]
            elif interval_policy == "keep":
                pass
            else:
                raise ValueError("interval_policy must be 'prorate', 'keep_start', 'exclude' or 'keep'")
    return ev.drop(columns=["_basis"]).reset_index(drop=True)


def aggregate_events(
    selected: pd.DataFrame,
    aggregations: Sequence[str] = ("last", "min", "max", "mean", "count"),
    variables: Optional[Iterable[str]] = None,
    time_unit: str = "hours",
) -> pd.DataFrame:
    """Pivot selected events into one row per entity and one column per variable and statistic."""
    bad = [a for a in aggregations if a not in SUPPORTED_AGGREGATIONS]
    if bad:
        raise ValueError(f"Unsupported aggregations {bad}; choose from {SUPPORTED_AGGREGATIONS}")
    ev = selected
    if variables is not None:
        ev = ev.loc[ev["variable"].isin(set(map(str, variables)))]
    if ev.empty:
        return pd.DataFrame(index=pd.Index([], name="entity_id"))
    ev = ev.copy()
    ev["_num"] = pd.to_numeric(ev["value"], errors="coerce")
    order_time = ev["available_time"].where(ev["available_time"].notna(), ev["event_time"])
    ev["_t"] = order_time
    ev = ev.sort_values(["entity_id", "variable", "_t"])
    g = ev.groupby(["entity_id", "variable"], sort=False)
    parts = {}
    for agg in aggregations:
        if agg == "count":
            parts[agg] = g["value"].size()
        elif agg == "last":
            parts[agg] = g["_num"].last()
        elif agg == "first":
            parts[agg] = g["_num"].first()
        elif agg == "time_since_last":
            last_t = g["_t"].max()
            ref = ev.groupby(["entity_id", "variable"], sort=False)["index_time"].first() if "index_time" in ev.columns else None
            if ref is None:
                continue
            delta = ref - last_t
            if pd.api.types.is_timedelta64_dtype(delta):
                div = 3600.0 if time_unit == "hours" else 60.0
                delta = delta.dt.total_seconds() / div
            parts[agg] = delta
        else:
            parts[agg] = getattr(g["_num"], agg)()
    frames = []
    for agg, s in parts.items():
        wide = s.unstack("variable")
        wide.columns = [f"{v}__{agg}" for v in wide.columns]
        frames.append(wide)
    out = pd.concat(frames, axis=1)
    out.index.name = "entity_id"
    return out.sort_index(axis=1)


def build_window_features(
    events: pd.DataFrame,
    index_times: Union[pd.DataFrame, pd.Series],
    window=None,
    aggregations: Sequence[str] = ("last", "min", "max", "mean", "count"),
    variables: Optional[Iterable[str]] = None,
    time_basis: str = "available",
    cutoff: str = "index",
    interval_policy: str = "prorate",
    gap=None,
    return_provenance: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Select events up to each prediction time and aggregate them.

    Returns a wide feature table indexed by ``entity_id``. With
    ``return_provenance=True`` also returns the event rows that were used, with
    a ``feature_prefix`` column, which can be passed to the audit as
    ``events`` to document exactly what each feature was built from.
    """
    sel = select_events(
        events, index_times, window=window, time_basis=time_basis,
        cutoff=cutoff, interval_policy=interval_policy, gap=gap,
    )
    feats = aggregate_events(sel, aggregations=aggregations, variables=variables)
    idx = _as_index_frame(index_times)
    feats = feats.reindex(pd.Index(idx["entity_id"].values, name="entity_id"))
    if return_provenance:
        prov = sel.copy()
        prov["feature_prefix"] = prov["variable"].astype(str) + "__"
        return feats, prov
    return feats


def make_feature_builder(
    window=None,
    aggregations: Sequence[str] = ("last", "min", "max", "mean", "count"),
    variables: Optional[Iterable[str]] = None,
    time_basis: str = "available",
    interval_policy: str = "prorate",
    gap=None,
):
    """Return ``fn(events, index_times) -> features`` for the truncation check."""

    def _builder(events: pd.DataFrame, index_times: pd.DataFrame) -> pd.DataFrame:
        return build_window_features(
            events, index_times, window=window, aggregations=aggregations, variables=variables,
            time_basis=time_basis, cutoff="index", interval_policy=interval_policy, gap=gap,
        )

    return _builder
