"""Quantify how much a leak inflates reported performance.

Typical use: train the same model on a leakage-free pipeline and on a pipeline
with one shortcut switched on, score both on the same evaluation rows, then
call :func:`leakage_impact`.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .metrics import METRICS, bootstrap_ci, paired_bootstrap_difference

DEFAULT_METRICS = ("auroc", "auprc", "brier", "calibration_slope", "calibration_intercept", "ece")


def leakage_impact(
    y_true,
    clean_scores,
    leaky_scores,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_boot: int = 1000,
    alpha: float = 0.05,
    groups: Optional[Sequence] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare a clean and a leaky pipeline on the same evaluation rows.

    Returns one row per metric with both estimates, their bootstrap intervals,
    the paired difference (leaky minus clean) with its interval and a
    two-sided bootstrap p-value. Pass ``groups`` (for example patient
    identifiers) to resample clusters rather than rows.
    """
    rows = []
    for m in metrics:
        fn = METRICS[m]
        c = bootstrap_ci(fn, y_true, clean_scores, n_boot=n_boot, alpha=alpha, random_state=random_state, groups=groups)
        l = bootstrap_ci(fn, y_true, leaky_scores, n_boot=n_boot, alpha=alpha, random_state=random_state, groups=groups)
        d = paired_bootstrap_difference(fn, y_true, clean_scores, leaky_scores, n_boot=n_boot, alpha=alpha, random_state=random_state, groups=groups)
        rows.append({
            "metric": m,
            "clean": c["estimate"], "clean_lower": c["lower"], "clean_upper": c["upper"],
            "leaky": l["estimate"], "leaky_lower": l["lower"], "leaky_upper": l["upper"],
            "difference": d["difference"], "diff_lower": d["lower"], "diff_upper": d["upper"],
            "p_value": d["p_two_sided"],
        })
    return pd.DataFrame(rows)


def leakage_impact_table(
    y_true,
    clean_scores,
    leaky_variants: Mapping[str, Sequence[float]],
    metrics: Sequence[str] = ("auroc", "brier", "calibration_slope"),
    n_boot: int = 1000,
    groups: Optional[Sequence] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run :func:`leakage_impact` for several leaky variants against one clean baseline."""
    frames = []
    for name, scores in leaky_variants.items():
        df = leakage_impact(y_true, clean_scores, scores, metrics=metrics, n_boot=n_boot, groups=groups, random_state=random_state)
        df.insert(0, "variant", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def inflation_summary(table: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """One line per variant: how far ``metric`` moved and whether the interval excludes zero."""
    t = table.loc[table["metric"] == metric].copy()
    t["significant"] = (t["diff_lower"] > 0) | (t["diff_upper"] < 0)
    cols = ["variant", "clean", "leaky", "difference", "diff_lower", "diff_upper", "p_value", "significant"]
    return t[[c for c in cols if c in t.columns]].sort_values("difference", ascending=False).reset_index(drop=True)
