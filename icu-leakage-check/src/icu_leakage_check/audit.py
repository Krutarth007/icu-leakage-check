"""Run a set of checks against one prediction dataset."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._version import __version__
from .checks import CHECK_REGISTRY, DEFAULT_CHECKS, Check
from .context import AuditContext, build_context
from .findings import AuditResult, Finding


class LeakageAudit:
    """Audit a prediction dataset for data leakage.

    ``data`` holds one row per prediction instance: the features the model
    sees, the label, the split assignment and identifiers. Every other input
    is optional; each check runs when its inputs are present and is reported
    as skipped otherwise, so the report also shows what was *not* verified.

    Parameters
    ----------
    data:
        DataFrame with one row per prediction instance.
    label:
        Outcome column (binary 0/1 for the outcome-based checks).
    split:
        Column name or array assigning rows to train / valid / test.
        Common synonyms (``development``, ``holdout``, ``external`` ...) are
        recognised.
    patient_id, encounter_id, stay_id, site_id:
        Identifier columns. With ``dataset`` set these default to the
        dataset's own identifiers when those columns exist.
    index_time:
        Prediction time of each row (datetime, or minutes for eICU-CRD).
    outcome_time, eligibility_time:
        Time the outcome occurred and time the row became eligible.
    calendar_time:
        Calendar variable for temporal ordering (``anchor_year_group`` for
        MIMIC-IV, ``hospitaldischargeyear`` for eICU-CRD).
    dataset:
        ``"mimic-iv"``, ``"eicu"`` or ``None`` for any other source.
    feature_columns:
        Features the model uses. Defaults to every column that is not an
        identifier, time, label, split or score.
    feature_sources:
        Mapping from feature to ``"table.column"`` (or ``"table"``). Keys
        ending in ``*`` match prefixes, for example ``{"lab_*": "lab"}``.
        Tag sources with ``prior`` (``"diagnoses_icd.prior"``) when they come
        from earlier encounters.
    events:
        Provenance log: the canonical event rows each feature was computed
        from (see :mod:`icu_leakage_check.events`).
    raw_events, feature_builder:
        Full event log and the function that turns events into features,
        used to test whether features change when future events are removed.
    preprocessor:
        Fitted scikit-learn transformer or Pipeline to inspect.
    scores, threshold:
        Model scores (column or array) and the reported decision threshold.
    split_strategy:
        What the split claims to be: ``"random"``, ``"patient"``,
        ``"temporal"``, ``"site"`` or ``"cross_validation"``.
    declarations:
        Practices that cannot be verified from data, as True/False; see
        :data:`icu_leakage_check.checks.DECLARATION_ITEMS`.
    options:
        Per-check settings as ``{"check_id.option": value}``, for example
        ``{"single_feature_dominance.high_auroc": 0.85}``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        label: Optional[str] = None,
        split=None,
        *,
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
        exclude_columns: Sequence[str] = (),
    ):
        from .schemas import get_schema

        schema = get_schema(dataset)
        if schema is not None:
            cols = set(data.columns)
            patient_id = patient_id or (schema.patient_id if schema.patient_id in cols else None)
            encounter_id = encounter_id or (schema.encounter_id if schema.encounter_id in cols else None)
            stay_id = stay_id or (schema.stay_id if schema.stay_id in cols else None)
            site_id = site_id or (schema.site_id if schema.site_id and schema.site_id in cols else None)
        self.context: AuditContext = build_context(
            data, label=label, split=split, patient_id=patient_id, encounter_id=encounter_id,
            stay_id=stay_id, site_id=site_id, index_time=index_time, outcome_time=outcome_time,
            eligibility_time=eligibility_time, calendar_time=calendar_time, dataset=schema,
            feature_columns=feature_columns, feature_sources=feature_sources, events=events,
            event_entity=event_entity, raw_events=raw_events, feature_builder=feature_builder,
            preprocessor=preprocessor, preprocessor_columns=preprocessor_columns, scores=scores,
            threshold=threshold, split_strategy=split_strategy, declarations=declarations,
            options=options, extra_exclude=exclude_columns,
        )

    # ------------------------------------------------------------------ run
    def run(
        self,
        checks: Optional[Iterable[Union[str, Check, type]]] = None,
        skip: Iterable[str] = (),
    ) -> AuditResult:
        """Run the checks and return an :class:`AuditResult`."""
        instances = self._resolve(checks)
        skip = set(skip)
        findings: List[Finding] = []
        for chk in instances:
            if chk.check_id in skip:
                continue
            findings.extend(chk.execute(self.context))
        return AuditResult(findings=findings, metadata=self._metadata(instances))

    def _resolve(self, checks) -> List[Check]:
        if checks is None:
            return [c() for c in DEFAULT_CHECKS]
        out = []
        for c in checks:
            if isinstance(c, Check):
                out.append(c)
            elif isinstance(c, type) and issubclass(c, Check):
                out.append(c())
            elif isinstance(c, str):
                if c not in CHECK_REGISTRY:
                    raise KeyError(f"Unknown check {c!r}; available: {sorted(CHECK_REGISTRY)}")
                out.append(CHECK_REGISTRY[c]())
            else:
                raise TypeError(f"Cannot interpret check {c!r}")
        return out

    def _metadata(self, instances: List[Check]) -> Dict[str, Any]:
        ctx = self.context
        splits = {}
        y = ctx.label_values()
        for s in ctx.split_names():
            m = ctx.split == s
            entry = {"rows": int(m.sum())}
            if y is not None and ctx.is_binary_label():
                entry["prevalence"] = round(float(y[m].mean()), 4)
            splits[s] = entry
        return {
            "package": "icu-leakage-check",
            "version": __version__,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset": ctx.schema.name if ctx.schema else None,
            "rows": int(len(ctx.data)),
            "features": int(len(ctx.feature_columns)),
            "label": ctx.label,
            "split_strategy": ctx.split_strategy,
            "splits": splits,
            "identifiers": {
                "patient_id": ctx.patient_id, "encounter_id": ctx.encounter_id,
                "stay_id": ctx.stay_id, "site_id": ctx.site_id,
            },
            "inputs": {
                "index_time": ctx.index_time is not None,
                "outcome_time": ctx.outcome_time is not None,
                "events": ctx.events is not None,
                "feature_builder": ctx.feature_builder is not None,
                "preprocessor": ctx.preprocessor is not None,
                "feature_sources": len(ctx.feature_sources),
                "declarations": len(ctx.declarations),
            },
            "checks": [c.check_id for c in instances],
        }

    # --------------------------------------------------------------- config
    @classmethod
    def from_config(cls, config: Union[str, Path, Mapping[str, Any]], data: Optional[pd.DataFrame] = None, **overrides) -> "LeakageAudit":
        """Build an audit from a JSON file or dict.

        Keys mirror the constructor arguments. ``data`` and ``events`` may be
        file paths (CSV or Parquet). Example::

            {"data": "features.parquet", "label": "died", "split": "split",
             "dataset": "eicu", "index_time": "index_offset",
             "feature_sources": {"apache_score": "apachePatientResult.apachescore"}}
        """
        if isinstance(config, (str, Path)):
            base = Path(config).parent
            with open(config, "r", encoding="utf-8") as fh:
                cfg = dict(json.load(fh))
        else:
            base = Path(".")
            cfg = dict(config)
        cfg.update(overrides)
        if data is None:
            if "data" not in cfg:
                raise KeyError("config needs a 'data' path or pass data=")
            data = _read_any(_resolve_path(cfg.pop("data"), base))
        else:
            cfg.pop("data", None)
        for key in ("events", "raw_events"):
            if isinstance(cfg.get(key), str):
                cfg[key] = _read_any(_resolve_path(cfg[key], base))
        for key in ("html", "json", "markdown", "fail_on"):
            cfg.pop(key, None)
        return cls(data, **cfg)


def _resolve_path(p: str, base: Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (base / path)


def _read_any(path: Union[str, Path]) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, low_memory=False)
    if p.endswith((".pkl", ".pickle")):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported file type: {path}")


def audit(data: pd.DataFrame, label: Optional[str] = None, split=None, **kwargs) -> AuditResult:
    """One-call shortcut: ``audit(df, "label", "split", dataset="mimic-iv")``."""
    checks = kwargs.pop("checks", None)
    skip = kwargs.pop("skip", ())
    return LeakageAudit(data, label, split, **kwargs).run(checks=checks, skip=skip)


def audit_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    label: Optional[str] = None,
    valid: Optional[pd.DataFrame] = None,
    **kwargs,
) -> AuditResult:
    """Audit separate train / (valid) / test frames without combining them yourself."""
    parts = [train.assign(__split__="train"), test.assign(__split__="test")]
    if valid is not None:
        parts.insert(1, valid.assign(__split__="valid"))
    combined = pd.concat(parts, ignore_index=True)
    return audit(combined, label, "__split__", **kwargs)
