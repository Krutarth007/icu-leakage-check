"""icu-leakage-check: detect, explain and quantify data leakage in clinical prediction pipelines.

Quick start::

    from icu_leakage_check import audit

    result = audit(features, label="died", split="split", dataset="eicu",
                   index_time="index_offset")
    print(result.summary())
    result.to_html("leakage_report.html")
"""

from ._version import __version__
from .audit import LeakageAudit, audit, audit_split
from .checks import DEFAULT_CHECKS, Check, list_checks, register
from .compare import inflation_summary, leakage_impact, leakage_impact_table
from .events import attach_entity, events_from_table, index_times_from_start
from .features import aggregate_events, build_window_features, make_feature_builder, select_events
from .findings import AuditResult, Finding, Severity, merge_results
from .io import convert_to_parquet, find_table, list_tables, query_table, read_table
from .metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    brier,
    calibration_slope_intercept,
    evaluate,
    expected_calibration_error,
    paired_bootstrap_difference,
)
from .schemas import EICU, MIMIC_IV, find_proxy_items, flag_proxy_codes, get_schema
from .splits import calendar_start_year, describe_split, group_split, site_split, temporal_split
from .taxonomy import LEAKAGE_TYPES, taxonomy_table

__all__ = [
    "AuditResult",
    "Check",
    "DEFAULT_CHECKS",
    "EICU",
    "Finding",
    "LEAKAGE_TYPES",
    "LeakageAudit",
    "MIMIC_IV",
    "Severity",
    "__version__",
    "aggregate_events",
    "attach_entity",
    "audit",
    "audit_split",
    "auprc",
    "auroc",
    "bootstrap_ci",
    "brier",
    "build_window_features",
    "calendar_start_year",
    "calibration_slope_intercept",
    "convert_to_parquet",
    "describe_split",
    "evaluate",
    "events_from_table",
    "expected_calibration_error",
    "find_proxy_items",
    "find_table",
    "flag_proxy_codes",
    "get_schema",
    "group_split",
    "index_times_from_start",
    "inflation_summary",
    "leakage_impact",
    "leakage_impact_table",
    "list_checks",
    "list_tables",
    "make_feature_builder",
    "merge_results",
    "paired_bootstrap_difference",
    "query_table",
    "read_table",
    "register",
    "select_events",
    "site_split",
    "taxonomy_table",
    "temporal_split",
]
