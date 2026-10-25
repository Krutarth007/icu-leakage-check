"""Temporal checks: was every piece of information available at prediction time?"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..findings import Finding, Severity, Status
from .base import Check, fraction, need, register


def _same_kind(a: pd.Series, b: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(a) == pd.api.types.is_datetime64_any_dtype(b)


def _hours(delta: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(delta):
        return delta.dt.total_seconds() / 3600.0
    return delta.astype(float) / 60.0  # minute offsets


def _attach_index(ctx, events: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "index_time" in events.columns:
        return events
    key = ctx.entity_key()
    idx = ctx.data[[key, ctx.index_time]].dropna()
    per = idx.groupby(key)[ctx.index_time].nunique()
    if (per > 1).any():
        return None
    idx = idx.drop_duplicates(key).rename(columns={key: "entity_id", ctx.index_time: "index_time"})
    return events.merge(idx, on="entity_id", how="inner")


def _feature_key(events: pd.DataFrame) -> pd.Series:
    for col in ("feature", "feature_prefix", "variable", "source"):
        if col in events.columns:
            return events[col].astype(str)
    return pd.Series("unknown", index=events.index)


def _map_to_features(keys, features: List[str]) -> Dict[str, List[str]]:
    out = {}
    fset = set(features)
    for k in keys:
        if k in fset:
            out[k] = [k]
            continue
        prefix = k if k.endswith("__") else k + "__"
        matched = [f for f in features if f.startswith(prefix)]
        out[k] = matched if matched else [k]
    return out


@register
class EventTimingCheck(Check):
    check_id = "event_timing"
    leakage_type = "FUTURE_EVENT"
    title = "Timing of the events used to build features"
    description = (
        "Uses an event-level provenance log (the rows each feature was computed from) to find events "
        "recorded after the prediction time, results not yet available, intervals running past the "
        "prediction time and undated sources."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.events is not None, "events (provenance of the features)", out)
        need(ctx.index_time is not None or (ctx.events is not None and "index_time" in ctx.events.columns), "index_time", out)
        need(ctx.entity_key() is not None or (ctx.events is not None and "index_time" in ctx.events.columns), "an entity identifier", out)
        if ctx.events is not None:
            for c in ("entity_id", "event_time"):
                need(c in ctx.events.columns, f"events column '{c}'", out)
        return out

    def run(self, ctx):
        ev = _attach_index(ctx, ctx.events)
        if ev is None:
            return [self.skipped(["events must include index_time when an entity has several prediction times"])]
        if ev.empty:
            return [self.skipped(["no events matched the prediction instances"])]
        ev = ev.reset_index(drop=True)
        if not _same_kind(ev["event_time"], ev["index_time"]):
            raise TypeError("event_time and index_time must both be datetimes or both be numeric offsets")
        key = _feature_key(ev)
        idx_t = ev["index_time"]
        et = ev["event_time"]
        at = ev["available_time"] if "available_time" in ev.columns else pd.Series(pd.NaT if pd.api.types.is_datetime64_any_dtype(et) else np.nan, index=ev.index)
        end = ev["end_time"] if "end_time" in ev.columns else None

        undated = et.isna() & at.isna()
        future = et.notna() & (et > idx_t)
        lag = et.notna() & (et <= idx_t) & at.notna() & (at > idx_t)
        crossing = pd.Series(False, index=ev.index)
        if end is not None:
            crossing = et.notna() & (et <= idx_t) & end.notna() & (end > idx_t)
        post_outcome = pd.Series(False, index=ev.index)
        if ctx.outcome_time and "outcome_time" not in ev.columns:
            k = ctx.entity_key()
            ot = ctx.data[[k, ctx.outcome_time]].dropna().drop_duplicates(k).rename(columns={k: "entity_id", ctx.outcome_time: "_ot"})
            ev = ev.merge(ot, on="entity_id", how="left")
            if _same_kind(ev["_ot"], et):
                post_outcome = et.notna() & ev["_ot"].notna() & (et >= ev["_ot"].values)

        n_events = len(ev)
        findings: List[Finding] = []
        specs = [
            (future, "FUTURE_EVENT", Severity.CRITICAL, "events recorded after the prediction time"),
            (lag, "AVAILABILITY_LAG", Severity.HIGH, "events charted before the prediction time but only available after it"),
            (crossing, "INTERVAL_CROSSING", Severity.MEDIUM, "interval events still running at the prediction time"),
            (undated, "UNDATED_SOURCE", Severity.HIGH, "events without any time stamp"),
        ]
        feats = ctx.feature_columns
        for mask, lt, sev, label in specs:
            n = int(mask.sum())
            if n == 0:
                findings.append(self.passed(f"No {label}.", {"events_checked": n_events}, leakage_type=lt))
                continue
            by_key = key[mask].value_counts()
            ents = ev.loc[mask, "entity_id"].nunique()
            mapping = _map_to_features(by_key.index.tolist(), feats)
            affected = []
            for k in by_key.index:
                for f in mapping[k]:
                    if f not in affected:
                        affected.append(f)
            evidence = {
                "events_flagged": n,
                "events_checked": n_events,
                "fraction_of_events": round(fraction(n, n_events), 4),
                "entities_affected": int(ents),
                "events_by_source": {str(k): int(v) for k, v in by_key.head(25).items()},
                "affected_kind": "feature",
            }
            if lt == "FUTURE_EVENT":
                lead = _hours(et[mask] - idx_t[mask])
                evidence["median_hours_after_prediction"] = round(float(lead.median()), 2)
            if lt == "AVAILABILITY_LAG":
                delay = _hours(at[mask] - idx_t[mask])
                evidence["median_hours_until_available"] = round(float(delay.median()), 2)
            findings.append(self.finding(
                Status.FAIL, sev,
                f"{n:,} {label} ({fraction(n, n_events):.1%} of events, {ents:,} prediction units), feeding {len(affected)} features.",
                evidence=evidence, affected=affected, leakage_type=lt,
                title={
                    "FUTURE_EVENT": "Features use events after the prediction time",
                    "AVAILABILITY_LAG": "Features use results not yet available",
                    "INTERVAL_CROSSING": "Interval events cross the prediction time",
                    "UNDATED_SOURCE": "Features use undated events",
                }[lt],
            ))
        if post_outcome.any():
            n = int(post_outcome.sum())
            by_key = key[post_outcome].value_counts()
            mapping = _map_to_features(by_key.index.tolist(), feats)
            affected = sorted({f for k in by_key.index for f in mapping[k]})
            findings.append(self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"{n:,} events are recorded at or after the outcome itself.",
                evidence={"events_flagged": n, "events_by_source": {str(k): int(v) for k, v in by_key.head(25).items()}, "affected_kind": "feature"},
                affected=affected, leakage_type="FUTURE_EVENT", title="Features use events after the outcome",
            ))
        return findings


def _values_equal(a: pd.Series, b: pd.Series, rtol: float = 1e-6, atol: float = 1e-9) -> pd.Series:
    an = pd.to_numeric(a, errors="coerce")
    bn = pd.to_numeric(b, errors="coerce")
    both_nan = a.isna() & b.isna()
    numeric = an.notna() & bn.notna()
    eq = pd.Series(False, index=a.index)
    eq[both_nan] = True
    if numeric.any():
        eq[numeric] = np.isclose(an[numeric].to_numpy(float), bn[numeric].to_numpy(float), rtol=rtol, atol=atol)
    other = ~both_nan & ~numeric
    if other.any():
        eq[other] = a[other].astype(str).values == b[other].astype(str).values
    return eq


@register
class TruncationInvarianceCheck(Check):
    check_id = "truncation_invariance"
    leakage_type = "FUTURE_EVENT"
    title = "Features change when future events are removed"
    description = (
        "Rebuilds features with the user's own feature function from events truncated at the prediction "
        "time and compares them with features built from the full record. Any difference means the "
        "feature function reads information from after the prediction time."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.feature_builder is not None, "feature_builder", out)
        need(ctx.raw_events is not None, "raw_events", out)
        need(ctx.index_time is not None, "index_time", out)
        need(ctx.entity_key() is not None, "an entity identifier", out)
        return out

    def run(self, ctx):
        sample_size = self.opt(ctx, "sample_size", 2000)
        seed = self.opt(ctx, "random_state", 42)
        tolerance = self.opt(ctx, "max_changed_fraction", 0.0)
        key = ctx.entity_key()
        idx = ctx.data[[key, ctx.index_time]].dropna().drop_duplicates(key)
        idx = idx.rename(columns={key: "entity_id", ctx.index_time: "index_time"})
        if len(idx) > sample_size:
            idx = idx.sample(sample_size, random_state=seed)
        raw = ctx.raw_events.loc[ctx.raw_events["entity_id"].isin(set(idx["entity_id"]))]
        merged = raw.merge(idx, on="entity_id", how="inner")
        basis = merged["available_time"].where(merged["available_time"].notna(), merged["event_time"]) if "available_time" in merged.columns else merged["event_time"]
        truncated = merged.loc[basis.notna() & (basis <= merged["index_time"])].drop(columns=["index_time"])
        if "end_time" in truncated.columns:
            truncated = truncated.copy()
            h = truncated["entity_id"].map(dict(zip(idx["entity_id"], idx["index_time"])))
            running = truncated["end_time"].notna() & (truncated["end_time"] > h)
            truncated.loc[running, "end_time"] = h[running]

        full = ctx.feature_builder(raw, idx)
        trunc = ctx.feature_builder(truncated, idx)
        if not isinstance(full, pd.DataFrame) or not isinstance(trunc, pd.DataFrame):
            raise TypeError("feature_builder must return a DataFrame indexed by entity_id")
        full = full.reindex(pd.Index(idx["entity_id"].values))
        trunc = trunc.reindex(index=full.index, columns=full.columns)
        changed = {}
        for c in full.columns:
            eq = _values_equal(full[c].reset_index(drop=True), trunc[c].reset_index(drop=True))
            frac = 1.0 - float(eq.mean()) if len(eq) else 0.0
            if frac > tolerance:
                changed[c] = round(frac, 4)
        ev = {"entities_sampled": int(len(idx)), "features_compared": int(full.shape[1]), "affected_kind": "feature"}
        if changed:
            ev["changed_fraction_by_feature"] = dict(sorted(changed.items(), key=lambda kv: -kv[1])[:50])
            return [self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"{len(changed)} of {full.shape[1]} features change when events after the prediction time are removed.",
                evidence=ev, affected=list(ev["changed_fraction_by_feature"].keys()) + [c for c in changed if c not in ev["changed_fraction_by_feature"]],
            )]
        return [self.passed(f"All {full.shape[1]} features are identical with and without future events.", ev)]


@register
class OutcomeTimingCheck(Check):
    check_id = "outcome_timing"
    leakage_type = "POST_OUTCOME_INSTANCE"
    title = "Outcome already present at prediction time"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.outcome_time is not None, "outcome_time", out)
        need(ctx.index_time is not None, "index_time", out)
        return out

    def run(self, ctx):
        ot = ctx.data[ctx.outcome_time]
        it = ctx.data[ctx.index_time]
        if not _same_kind(ot, it):
            raise TypeError("outcome_time and index_time must both be datetimes or both be numeric offsets")
        has = ot.notna() & it.notna()
        if ctx.label:
            y = ctx.label_values()
            has &= y == 1
        n_pos = int(has.sum())
        if n_pos == 0:
            return [self.skipped(["positive rows with an outcome time"])]
        lead = _hours(ot[has] - it[has])
        already = int((lead <= 0).sum())
        min_gap = float(self.opt(ctx, "min_gap_hours", 0.0))
        near = int(((lead > 0) & (lead <= min_gap)).sum()) if min_gap > 0 else 0
        ev = {
            "positives_with_time": n_pos,
            "outcome_at_or_before_prediction": already,
            "fraction": round(fraction(already, n_pos), 4),
            "median_lead_hours": round(float(lead.median()), 2),
            "lead_hours_quantiles": {q: round(float(lead.quantile(q)), 2) for q in (0.1, 0.25, 0.5, 0.75)},
            "affected_kind": "row",
        }
        findings = []
        if already:
            findings.append(self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"{already} positive instances ({ev['fraction']:.1%}) already had the outcome at the prediction time.",
                evidence=ev, affected=[str(i) for i in ctx.data.index[has][lead.values <= 0][:10]],
            ))
        if near:
            ev2 = dict(ev, within_gap=near, min_gap_hours=min_gap)
            findings.append(self.finding(
                Status.WARN, Severity.MEDIUM,
                f"{near} positive instances have the outcome within {min_gap:g} hours of prediction, leaving little time to act.",
                evidence=ev2,
            ))
        if not findings:
            findings.append(self.passed("Every positive instance has its outcome after the prediction time.", ev))
        return findings


@register
class EligibilityTimingCheck(Check):
    check_id = "eligibility_timing"
    leakage_type = "COHORT_FUTURE_INFORMATION"
    title = "Cohort eligibility decided after prediction time"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.eligibility_time is not None, "eligibility_time", out)
        need(ctx.index_time is not None, "index_time", out)
        return out

    def run(self, ctx):
        el = ctx.data[ctx.eligibility_time]
        it = ctx.data[ctx.index_time]
        if not _same_kind(el, it):
            raise TypeError("eligibility_time and index_time must both be datetimes or both be numeric offsets")
        ok = el.notna() & it.notna()
        late = ok & (el > it)
        n = int(late.sum())
        ev = {"rows_checked": int(ok.sum()), "eligibility_after_prediction": n, "fraction": round(fraction(n, int(ok.sum())), 4), "affected_kind": "row"}
        if n:
            ev["median_hours_late"] = round(float(_hours(el[late] - it[late]).median()), 2)
            return [self.finding(
                Status.FAIL, Severity.HIGH,
                f"{n} rows ({ev['fraction']:.1%}) only qualified for the cohort after their prediction time.",
                evidence=ev, affected=[str(i) for i in ctx.data.index[late][:10]],
            )]
        return [self.passed("Eligibility was known at or before every prediction time.", ev)]
