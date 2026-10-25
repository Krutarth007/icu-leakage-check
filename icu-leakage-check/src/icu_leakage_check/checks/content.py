"""Feature content: does a feature carry outcome or hindsight information?"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..findings import Finding, Severity, Status
from ..metrics import auroc
from ..schemas.base import DATE_ONLY, FIRST_DAY, UNTIMED
from ..schemas.codes import (
    match_hindsight_name,
    match_outcome_name,
    match_proxy_code,
    match_proxy_feature_name,
    match_proxy_text,
)
from ..splits import TRAIN, is_text
from .base import Check, need, register

# Registry column names too generic to flag on name alone.
_GENERIC_NAMES = {
    "status", "description", "seq_num", "value", "organism", "antibiotic", "sensitivitylevel",
    "drg_type", "icd_code", "drg_code", "outtime", "dischargelocation",
}
_FIRST_DAY_NAME_PREFIXES = ("day1", "saps3", "apachescore", "acutephysiologyscore", "predicted", "predventdays", "predicted_")


def _prior_tagged(src: str) -> bool:
    s = (src or "").lower()
    return any(t in s for t in ("prior", "previous", "history", "pre_admission", "preadmission"))


class _Collector:
    def __init__(self):
        self.items: Dict[str, Dict[str, Tuple[Severity, str, str]]] = {}

    def add(self, lt: str, feature: str, severity: Severity, status: str, reason: str):
        cur = self.items.setdefault(lt, {})
        if feature not in cur or cur[feature][0] < severity:
            cur[feature] = (severity, status, reason)


@register
class HindsightFeatureCheck(Check):
    check_id = "hindsight_features"
    leakage_type = "HINDSIGHT_VARIABLE"
    title = "Hindsight and outcome variables among the features"
    description = (
        "Traces each feature to its source table and column (from feature_sources, or from the dataset "
        "registry by name) and flags outcome fields, discharge-time fields, undated billing sources and "
        "first-day summaries used before the first day is over."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(len(ctx.feature_columns) > 0, "feature columns", out)
        return out

    def _first_day_status(self, ctx, n_reason: str):
        schema = ctx.schema
        if ctx.index_time is None or schema is None or schema.time_kind != "minutes":
            return Severity.MEDIUM, Status.WARN, n_reason + " (prediction time unknown, so cannot confirm it is after the first 24 hours)"
        it = pd.to_numeric(ctx.data[ctx.index_time], errors="coerce")
        early = int((it < schema.first_day_minutes).sum())
        if early:
            return Severity.HIGH, Status.FAIL, n_reason + f"; {early} prediction times are before minute {schema.first_day_minutes:g}"
        return None, None, None

    def run(self, ctx):
        col = _Collector()
        schema = ctx.schema
        if ctx.label and ctx.label in ctx.feature_columns:
            col.add("OUTCOME_ENCODING", ctx.label, Severity.CRITICAL, Status.FAIL, "the label column is itself a feature")

        for f in ctx.feature_columns:
            table, column = ctx.source_of(f)
            src_raw = ctx.feature_sources.get(f, "")
            if table and schema is not None and schema.has_table(table):
                spec = schema.table(table)
                c = (column or "").lower()
                prior = _prior_tagged(src_raw)
                if c and c in spec.outcome_columns:
                    col.add("OUTCOME_ENCODING", f, Severity.CRITICAL, Status.FAIL, f"{spec.name}.{c} records the outcome")
                elif c and c in spec.hindsight_columns and not prior:
                    col.add("HINDSIGHT_VARIABLE", f, Severity.CRITICAL, Status.FAIL, f"{spec.name}.{c} is only known in hindsight")
                elif spec.timing == UNTIMED and not prior:
                    col.add("UNDATED_SOURCE", f, Severity.CRITICAL, Status.FAIL, f"{spec.name} has no time stamp and is finalised after discharge")
                elif spec.timing == DATE_ONLY and not prior:
                    col.add("UNDATED_SOURCE", f, Severity.MEDIUM, Status.WARN, f"{spec.name} has day resolution only; same-day rows may postdate the prediction")
                elif spec.timing == FIRST_DAY or (c and c in spec.first_day_columns):
                    sev, st, reason = self._first_day_status(ctx, f"{spec.name} summarises the first 24 hours")
                    if sev is not None:
                        col.add("FIRST_DAY_SUMMARY", f, sev, st, reason)
                continue

            name = str(f).lower()
            if schema is not None:
                roles = schema.column_roles(name)
                outcome_tables = [t for t, r in roles if r == "outcome"]
                hind_tables = [t for t, r in roles if r == "hindsight"]
                fd_tables = [t for t, r in roles if r == "first_day"]
                if any(r == "calendar" for _, r in roles):
                    continue
                if outcome_tables:
                    col.add("OUTCOME_ENCODING", f, Severity.CRITICAL, Status.FAIL, f"name matches outcome column in {', '.join(outcome_tables)}")
                    continue
                if hind_tables and name not in _GENERIC_NAMES:
                    col.add("HINDSIGHT_VARIABLE", f, Severity.HIGH, Status.FAIL, f"name matches hindsight column in {', '.join(hind_tables)}")
                    continue
                if fd_tables and name.startswith(_FIRST_DAY_NAME_PREFIXES):
                    sev, st, reason = self._first_day_status(ctx, f"name matches first-day column in {', '.join(fd_tables)}")
                    if sev is not None:
                        col.add("FIRST_DAY_SUMMARY", f, sev, st, reason)
                    continue
            lab = match_outcome_name(f)
            if lab:
                col.add("OUTCOME_ENCODING", f, Severity.HIGH, Status.WARN, f"name suggests {lab}")
                continue
            lab = match_hindsight_name(f)
            if lab:
                col.add("HINDSIGHT_VARIABLE", f, Severity.MEDIUM, Status.WARN, f"name suggests {lab}")

        findings: List[Finding] = []
        titles = {
            "OUTCOME_ENCODING": "Features that encode the outcome",
            "HINDSIGHT_VARIABLE": "Features only known in hindsight",
            "UNDATED_SOURCE": "Features from undated or day-resolution sources",
            "FIRST_DAY_SUMMARY": "First-day summaries used too early",
        }
        for lt, items in col.items.items():
            worst = max(v[0] for v in items.values())
            status = Status.FAIL if any(v[1] == Status.FAIL for v in items.values()) else Status.WARN
            ordered = sorted(items.items(), key=lambda kv: -kv[1][0])
            reasons = {k: v[2] for k, v in ordered}
            findings.append(self.finding(
                status, worst,
                f"{len(items)} feature(s) flagged: " + ", ".join(list(reasons)[:8]) + (" ..." if len(items) > 8 else ""),
                evidence={"reasons": reasons, "affected_kind": "feature"},
                affected=list(reasons), leakage_type=lt, title=titles[lt],
            ))
        if not findings:
            findings.append(self.passed(
                f"No hindsight, outcome, undated or early first-day variables recognised among {len(ctx.feature_columns)} features.",
                {"features_checked": len(ctx.feature_columns), "registry": schema.name if schema else None},
            ))
        return findings


@register
class OutcomeProxyCheck(Check):
    check_id = "outcome_proxies"
    leakage_type = "OUTCOME_PROXY"
    title = "Features that act as proxies for the outcome"
    description = (
        "Flags features that represent treatment-limitation, palliative, end-of-life or death-related "
        "documentation, from feature names, source tables and categorical values."
    )

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(len(ctx.feature_columns) > 0, "feature columns", out)
        return out

    def run(self, ctx):
        max_levels = self.opt(ctx, "max_levels", 200)
        reasons: Dict[str, str] = {}
        strong: Dict[str, bool] = {}
        schema = ctx.schema
        for f in ctx.feature_columns:
            table, column = ctx.source_of(f)
            if table and schema is not None and schema.has_table(table) and schema.table(table).whole_table_is_proxy:
                reasons[f] = f"derived from {schema.table(table).name}, which records end-of-life care"
                strong[f] = True
                continue
            lab = match_proxy_feature_name(f)
            if lab:
                reasons[f] = f"name refers to {lab}"
                pc = None
                for part in str(f).replace(".", "").upper().split("_"):
                    pc = pc or match_proxy_code(part)
                strong[f] = not (pc is not None and pc.strength == "context") and "Cardiac arrest" not in lab
                continue
            s = ctx.data[f]
            if is_text(s):
                uniq = pd.unique(s.dropna())
                if 0 < len(uniq) <= max_levels:
                    hits = [str(u) for u in uniq if match_proxy_text(str(u))]
                    if hits:
                        reasons[f] = "values include " + "; ".join(hits[:5])
                        strong[f] = True
        if not reasons:
            return [self.passed(f"No outcome proxies recognised among {len(ctx.feature_columns)} features.")]
        any_strong = any(strong.values())
        return [self.finding(
            Status.WARN,
            Severity.HIGH if any_strong else Severity.MEDIUM,
            f"{len(reasons)} feature(s) look like outcome proxies: " + ", ".join(list(reasons)[:8]) + (" ..." if len(reasons) > 8 else ""),
            evidence={"reasons": reasons, "affected_kind": "feature"},
            affected=list(reasons),
            recommendation=(
                "Confirm each proxy was documented before the prediction time. If it was, report it as a "
                "deliberate predictor and show performance without it; otherwise remove it."
            ),
        )]


def _binary_label_rows(ctx):
    y = ctx.label_values()
    rows = pd.Series(True, index=ctx.data.index)
    if ctx.has_split and TRAIN in ctx.split_names():
        rows = ctx.split == TRAIN
    rows &= y.notna()
    return y, rows


def _encode_for_auc(s: pd.Series, y: pd.Series, max_levels: int):
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if pd.api.types.is_datetime64_any_dtype(s):
        return None
    levels = s.astype(str).where(s.notna())
    if levels.nunique() > max_levels:
        return None
    means = y.groupby(levels).mean()
    return levels.map(means).astype(float)


@register
class FeatureDominanceCheck(Check):
    check_id = "single_feature_dominance"
    leakage_type = "SINGLE_FEATURE_DOMINANCE"
    title = "Single features that separate the outcome almost perfectly"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.label is not None, "label", out)
        need(ctx.is_binary_label(), "binary 0/1 label", out)
        need(len(ctx.feature_columns) > 0, "feature columns", out)
        return out

    def run(self, ctx):
        crit = float(self.opt(ctx, "critical_auroc", 0.98))
        high = float(self.opt(ctx, "high_auroc", 0.90))
        max_levels = int(self.opt(ctx, "max_levels", 20))
        min_rows = int(self.opt(ctx, "min_rows", 30))
        y, rows = _binary_label_rows(ctx)
        yr = y[rows]
        scores: Dict[str, float] = {}
        identical: List[str] = []
        for f in ctx.feature_columns:
            x = _encode_for_auc(ctx.data.loc[rows, f], yr, max_levels)
            if x is None:
                continue
            ok = x.notna()
            if ok.sum() < min_rows or yr[ok].nunique() < 2:
                continue
            a = auroc(yr[ok], x[ok])
            if np.isnan(a):
                continue
            scores[f] = max(a, 1 - a)
            if ok.mean() > 0.99 and set(pd.unique(x[ok])) <= {0.0, 1.0}:
                agree = (x[ok] == yr[ok]).mean()
                if agree >= 0.995 or agree <= 0.005:
                    identical.append(f)
        ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
        top = {k: round(v, 4) for k, v in list(ranked.items())[:15]}
        ev = {"features_scored": len(scores), "top_univariate_auroc": top, "rows_used": int(rows.sum()), "affected_kind": "feature"}
        findings: List[Finding] = []
        if identical:
            findings.append(self.finding(
                Status.FAIL, Severity.CRITICAL,
                f"{len(identical)} feature(s) are identical to the label (or its inverse): {', '.join(identical[:8])}.",
                evidence=ev, affected=identical, leakage_type="OUTCOME_ENCODING",
                title="Features identical to the label",
            ))
        critical = [f for f, v in ranked.items() if v >= crit and f not in identical]
        warn = [f for f, v in ranked.items() if high <= v < crit and f not in identical]
        if critical:
            findings.append(self.finding(
                Status.FAIL, Severity.HIGH,
                f"{len(critical)} feature(s) reach a univariate AUROC of at least {crit}: "
                + ", ".join(f"{f} ({ranked[f]:.3f})" for f in critical[:6]),
                evidence=ev, affected=critical,
            ))
        if warn:
            findings.append(self.finding(
                Status.WARN, Severity.MEDIUM,
                f"{len(warn)} feature(s) have a univariate AUROC between {high} and {crit}: "
                + ", ".join(f"{f} ({ranked[f]:.3f})" for f in warn[:6]),
                evidence=ev, affected=warn,
            ))
        if not findings:
            best = next(iter(top.items()), (None, None))
            findings.append(self.passed(
                f"No single feature exceeds a univariate AUROC of {high}"
                + (f" (highest: {best[0]} at {best[1]:.3f})." if best[0] else "."),
                ev,
            ))
        return findings


@register
class MissingnessProxyCheck(Check):
    check_id = "missingness_proxy"
    leakage_type = "MISSINGNESS_PROXY"
    title = "Missingness patterns that predict the outcome"

    def missing_inputs(self, ctx):
        out: List[str] = []
        need(ctx.label is not None, "label", out)
        need(ctx.is_binary_label(), "binary 0/1 label", out)
        need(len(ctx.feature_columns) > 0, "feature columns", out)
        return out

    def run(self, ctx):
        high = float(self.opt(ctx, "high_auroc", 0.92))
        warn_at = float(self.opt(ctx, "warn_auroc", 0.80))
        min_frac = float(self.opt(ctx, "min_missing_fraction", 0.01))
        y, rows = _binary_label_rows(ctx)
        yr = y[rows]
        scores: Dict[str, float] = {}
        rates: Dict[str, float] = {}
        for f in ctx.feature_columns:
            miss = ctx.data.loc[rows, f].isna().astype(float)
            rate = miss.mean()
            if rate < min_frac or rate > 1 - min_frac:
                continue
            a = auroc(yr, miss)
            if np.isnan(a):
                continue
            scores[f] = max(a, 1 - a)
            rates[f] = rate
        ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
        top = {k: {"auroc": round(v, 4), "missing_rate": round(rates[k], 4)} for k, v in list(ranked.items())[:15]}
        ev = {"features_scored": len(scores), "top_missingness_auroc": top, "affected_kind": "feature"}
        severe = [f for f, v in ranked.items() if v >= high]
        mild = [f for f, v in ranked.items() if warn_at <= v < high]
        findings: List[Finding] = []
        if severe:
            findings.append(self.finding(
                Status.FAIL, Severity.HIGH,
                f"Whether {len(severe)} feature(s) are missing separates outcomes with AUROC at least {high}: "
                + ", ".join(f"{f} ({ranked[f]:.3f})" for f in severe[:6]),
                evidence=ev, affected=severe,
            ))
        if mild:
            findings.append(self.finding(
                Status.WARN, Severity.MEDIUM,
                f"Missingness of {len(mild)} feature(s) is strongly associated with the outcome (AUROC {warn_at} to {high}).",
                evidence=ev, affected=mild,
            ))
        if not findings:
            findings.append(self.passed(f"No missingness indicator reaches an AUROC of {warn_at}.", ev))
        return findings
