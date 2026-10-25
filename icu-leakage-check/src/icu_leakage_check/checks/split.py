"""Split integrity: who and what ends up in which split."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..findings import Finding, Severity, Status
from ..splits import TEST, TRAIN, calendar_start_year, is_text
from .base import Check, fraction, need, pairs, register


@register
class EntityOverlapCheck(Check):
    check_id = "entity_overlap"
    leakage_type = "ENTITY_OVERLAP"
    title = "Patients or encounters shared between splits"
    description = "Each patient, admission and stay should belong to exactly one split."

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.has_split, "split", out)
        need(bool(ctx.patient_id or ctx.encounter_id or ctx.stay_id), "patient_id, encounter_id or stay_id", out)
        need(len(ctx.split_names()) >= 2, "at least two splits", out)
        return out

    def run(self, ctx):
        findings: List[Finding] = []
        levels = [
            ("stay", ctx.stay_id, Severity.CRITICAL),
            ("encounter", ctx.encounter_id, Severity.CRITICAL),
            ("patient", ctx.patient_id, Severity.HIGH),
        ]
        names = ctx.split_names()
        any_fail = False
        evidence = {}
        for level, col, sev in levels:
            if not col:
                continue
            ids = {s: set(ctx.rows(s)[col].dropna().unique()) for s in names}
            for a, b in pairs(names):
                shared = ids[a] & ids[b]
                if not shared:
                    continue
                any_fail = True
                eval_split = b if a == TRAIN else b
                eval_rows = ctx.rows(eval_split)
                n_rows = int(eval_rows[col].isin(shared).sum())
                ev = {
                    "level": level,
                    "column": col,
                    "splits": [a, b],
                    "shared_ids": len(shared),
                    f"rows_in_{eval_split}_affected": n_rows,
                    f"fraction_of_{eval_split}_rows": round(fraction(n_rows, len(eval_rows)), 4),
                    "example_ids": [str(x) for x in sorted(shared, key=str)[:10]],
                    "affected_kind": "id",
                }
                findings.append(self.finding(
                    Status.FAIL, sev,
                    f"{len(shared):,} {level} identifiers ({col}) appear in both '{a}' and '{b}'; "
                    f"{n_rows:,} rows ({ev[f'fraction_of_{eval_split}_rows']:.1%}) of '{eval_split}' are affected.",
                    evidence=ev,
                    affected=ev["example_ids"],
                ))
            evidence[level] = {s: len(v) for s, v in ids.items()}
        if not any_fail:
            findings.append(self.passed("No identifier is shared between splits.", {"distinct_ids_per_split": evidence}))
        return findings


@register
class SiteOverlapCheck(Check):
    check_id = "site_overlap"
    leakage_type = "SITE_OVERLAP"
    title = "Sites shared between training and evaluation"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.has_split, "split", out)
        need(ctx.site_id is not None, "site_id", out)
        return out

    def run(self, ctx):
        names = ctx.split_names()
        ids = {s: set(ctx.rows(s)[ctx.site_id].dropna().unique()) for s in names}
        train = ids.get(TRAIN, set())
        findings = []
        for s in ctx.evaluation_splits():
            shared = train & ids[s]
            ev = {
                "train_sites": len(train),
                f"{s}_sites": len(ids[s]),
                "shared_sites": len(shared),
                "example_sites": [str(x) for x in sorted(shared, key=str)[:10]],
                "affected_kind": "id",
            }
            if shared and ctx.split_strategy == "site":
                findings.append(self.finding(
                    Status.FAIL, Severity.CRITICAL,
                    f"Split is described as site-held-out but {len(shared)} of {len(ids[s])} sites in '{s}' also appear in training.",
                    evidence=ev, affected=ev["example_sites"],
                ))
            elif shared:
                findings.append(self.passed(
                    f"'{s}' shares {len(shared)} sites with training, so it measures internal rather than external performance.",
                    ev,
                ))
            else:
                findings.append(self.passed(f"'{s}' contains only sites unseen in training.", ev))
        return findings


def _row_hashes(df: pd.DataFrame, decimals: int) -> pd.Series:
    num = df.select_dtypes(include=[np.number, "bool"]).astype(float).round(decimals)
    other = df.drop(columns=num.columns)
    combined = pd.concat([num, other.astype(str)], axis=1) if len(other.columns) else num
    return pd.util.hash_pandas_object(combined, index=False)


@register
class DuplicateRecordCheck(Check):
    check_id = "duplicate_records"
    leakage_type = "DUPLICATE_RECORD"
    title = "Identical feature vectors in training and evaluation"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.has_split and TRAIN in ctx.split_names(), "split with a training part", out)
        need(len(ctx.evaluation_splits()) > 0, "an evaluation split", out)
        need(len(ctx.feature_columns) >= 2, "at least two feature columns", out)
        return out

    def run(self, ctx):
        decimals = self.opt(ctx, "decimals", 6)
        min_non_null = self.opt(ctx, "min_non_null", 3)
        fail_fraction = self.opt(ctx, "fail_fraction", 0.01)
        X = ctx.data[ctx.feature_columns]
        informative = X.notna().sum(axis=1) >= min_non_null
        h = _row_hashes(X, decimals)
        train_mask = (ctx.split == TRAIN) & informative
        train_hashes = set(h[train_mask])
        y = ctx.label_values()
        findings = []
        for s in ctx.evaluation_splits():
            m = (ctx.split == s) & informative
            dup = m & h.isin(train_hashes)
            n_dup = int(dup.sum())
            n_eval = int((ctx.split == s).sum())
            frac = fraction(n_dup, n_eval)
            ev = {"duplicates": n_dup, f"{s}_rows": n_eval, "fraction": round(frac, 4), "affected_kind": "row"}
            if y is not None and n_dup:
                train_lab = pd.DataFrame({"h": h[train_mask], "y": y[train_mask]}).groupby("h")["y"].mean()
                agree = (y[dup].round() == h[dup].map(train_lab).round()).mean()
                ev["label_agreement_with_training_copy"] = round(float(agree), 3)
            if n_dup and frac >= fail_fraction:
                sev = Severity.HIGH if ev.get("label_agreement_with_training_copy", 1.0) >= 0.9 else Severity.MEDIUM
                findings.append(self.finding(
                    Status.FAIL, sev,
                    f"{n_dup} rows ({frac:.1%}) of '{s}' have a feature vector identical to a training row.",
                    evidence=ev, affected=[str(i) for i in ctx.data.index[dup][:10]],
                ))
            elif n_dup:
                findings.append(self.finding(
                    Status.WARN, Severity.LOW,
                    f"{n_dup} rows ({frac:.2%}) of '{s}' duplicate a training row; below the {fail_fraction:.0%} threshold but worth checking.",
                    evidence=ev, affected=[str(i) for i in ctx.data.index[dup][:10]],
                ))
            else:
                findings.append(self.passed(f"No row of '{s}' duplicates a training row.", ev))
        return findings


@register
class TemporalOrderCheck(Check):
    check_id = "temporal_order"
    leakage_type = "SPLIT_ORDER"
    title = "Temporal order of training and evaluation periods"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.has_split and TRAIN in ctx.split_names(), "split with a training part", out)
        need(bool(ctx.calendar_time or ctx.index_time or ctx.split_strategy == "temporal"), "calendar_time or index_time", out)
        return out

    def _calendar(self, ctx):
        if ctx.calendar_time:
            s = ctx.data[ctx.calendar_time]
            if is_text(s):
                return calendar_start_year(s), ctx.calendar_time
            return s, ctx.calendar_time
        return ctx.index_series(), ctx.index_time

    def run(self, ctx):
        findings = []
        claimed = ctx.split_strategy == "temporal"
        is_mimic = ctx.schema is not None and ctx.schema.name == "mimic-iv"
        t, col = self._calendar(ctx)
        if is_mimic and claimed and (t is None or pd.api.types.is_datetime64_any_dtype(t)):
            findings.append(self.finding(
                Status.FAIL, Severity.HIGH,
                "Temporal validation in MIMIC-IV is ordered by shifted dates. Dates are shifted per patient, so they do not "
                "order patients in calendar time.",
                evidence={"time_column": col},
                recommendation="Order patients by anchor_year_group from the patients table and pass it as calendar_time.",
            ))
            return findings
        if t is None:
            return [self.skipped(["calendar_time"])]
        tr = t[ctx.split == TRAIN].dropna()
        if tr.empty:
            return [self.skipped(["training rows with a time value"])]
        for s in ctx.evaluation_splits():
            te = t[ctx.split == s].dropna()
            if te.empty:
                continue
            later = int((tr >= te.min()).sum())
            ev = {
                "time_column": col,
                "train_range": [str(tr.min()), str(tr.max())],
                f"{s}_range": [str(te.min()), str(te.max())],
                "train_rows_at_or_after_eval_start": later,
                "fraction_train_at_or_after_eval_start": round(fraction(later, len(tr)), 4),
            }
            if claimed and later > 0:
                findings.append(self.finding(
                    Status.FAIL, Severity.HIGH,
                    f"Split is described as temporal but {later} training rows ({ev['fraction_train_at_or_after_eval_start']:.1%}) "
                    f"fall at or after the start of '{s}', so the periods overlap.",
                    evidence=ev,
                ))
            elif claimed:
                findings.append(self.passed(f"Training precedes '{s}' in time.", ev))
            else:
                findings.append(self.passed(
                    f"Periods recorded for information; the split is not described as temporal ('{s}').", ev,
                ))
        return findings
