"""Discrimination and calibration metrics with bootstrap uncertainty.

Implemented with numpy only so results are identical across platforms.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

_EPS = 1e-12


def _clean(y_true, y_score):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    return y[ok], p[ok]


def auroc(y_true, y_score) -> float:
    y, p = _clean(y_true, y_score)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(y_true, y_score) -> float:
    """Average precision (step-wise area under the precision-recall curve)."""
    y, p = _clean(y_true, y_score)
    n1 = (y == 1).sum()
    if n1 == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    p_sorted = p[order]
    distinct = np.where(np.diff(p_sorted))[0]
    thresh_idx = np.r_[distinct, len(y_sorted) - 1]
    tps = np.cumsum(y_sorted)[thresh_idx]
    fps = 1 + thresh_idx - tps
    precision = tps / (tps + fps)
    recall = tps / n1
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def brier(y_true, y_score) -> float:
    y, p = _clean(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def _sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(z, -500, 500)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true, y_score, max_iter: int = 100) -> Dict[str, float]:
    """Logistic recalibration: slope, and calibration-in-the-large intercept.

    ``slope`` comes from ``logit(P(y=1)) = a + b * logit(p)``. ``intercept`` is
    the calibration-in-the-large estimate from the same model with the slope
    fixed at 1 (an offset), which is the conventional definition.
    """
    y, p = _clean(y_true, y_score)
    if len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan")}
    x = _logit(p)
    # Newton-Raphson for (a, b)
    beta = np.array([0.0, 1.0])
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        w = mu * (1 - mu) + _EPS
        grad = X.T @ (y - mu)
        hess = X.T @ (X * w[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    # Intercept with offset
    a = 0.0
    for _ in range(max_iter):
        mu = _sigmoid(a + x)
        g = np.sum(y - mu)
        h = np.sum(mu * (1 - mu)) + _EPS
        step = g / h
        a += step
        if abs(step) < 1e-10:
            break
    return {"slope": float(beta[1]), "intercept": float(a)}


def calibration_slope(y_true, y_score) -> float:
    return calibration_slope_intercept(y_true, y_score)["slope"]


def calibration_intercept(y_true, y_score) -> float:
    return calibration_slope_intercept(y_true, y_score)["intercept"]


def expected_calibration_error(y_true, y_score, n_bins: int = 10, strategy: str = "quantile") -> float:
    y, p = _clean(y_true, y_score)
    if len(y) == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    ece = 0.0
    for b in range(len(edges) - 1):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def observed_expected_ratio(y_true, y_score) -> float:
    y, p = _clean(y_true, y_score)
    if p.sum() == 0:
        return float("nan")
    return float(y.sum() / p.sum())


def net_benefit(y_true, y_score, threshold: float) -> float:
    y, p = _clean(y_true, y_score)
    n = len(y)
    if n == 0 or not 0 < threshold < 1:
        return float("nan")
    pred = p >= threshold
    tp = np.sum(pred & (y == 1))
    fp = np.sum(pred & (y == 0))
    return float(tp / n - fp / n * threshold / (1 - threshold))


METRICS: Dict[str, Callable] = {
    "auroc": auroc,
    "auprc": auprc,
    "brier": brier,
    "calibration_slope": calibration_slope,
    "calibration_intercept": calibration_intercept,
    "ece": expected_calibration_error,
    "oe_ratio": observed_expected_ratio,
}


def evaluate(y_true, y_score, metrics: Sequence[str] = ("auroc", "auprc", "brier", "calibration_slope", "calibration_intercept", "ece")) -> Dict[str, float]:
    return {m: METRICS[m](y_true, y_score) for m in metrics}


def bootstrap_ci(
    metric: Callable,
    y_true,
    y_score,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    groups: Optional[Sequence] = None,
) -> Dict[str, float]:
    """Percentile bootstrap interval. ``groups`` resamples clusters (e.g. patients)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(random_state)
    est = metric(y, p)
    stats = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            i = rng.integers(0, n, n)
            stats.append(metric(y[i], p[i]))
    else:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(uniq))]
        for _ in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            i = np.concatenate([members[k] for k in pick])
            stats.append(metric(y[i], p[i]))
    stats = np.asarray(stats, dtype=float)
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return {"estimate": est, "lower": float("nan"), "upper": float("nan")}
    return {
        "estimate": float(est),
        "lower": float(np.quantile(stats, alpha / 2)),
        "upper": float(np.quantile(stats, 1 - alpha / 2)),
    }


def paired_bootstrap_difference(
    metric: Callable,
    y_true,
    score_a,
    score_b,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
    groups: Optional[Sequence] = None,
) -> Dict[str, float]:
    """Bootstrap the difference ``metric(b) - metric(a)`` on the same resamples."""
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(random_state)
    est = metric(y, b) - metric(y, a)
    diffs = []
    if groups is None:
        n = len(y)
        draws = (rng.integers(0, n, n) for _ in range(n_boot))
    else:
        g = np.asarray(groups)
        uniq, inv = np.unique(g, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(uniq))]
        draws = (
            np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            for _ in range(n_boot)
        )
    for i in draws:
        diffs.append(metric(y[i], b[i]) - metric(y[i], a[i]))
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) == 0:
        return {"difference": float(est), "lower": float("nan"), "upper": float("nan"), "p_two_sided": float("nan")}
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "difference": float(est),
        "lower": float(np.quantile(diffs, alpha / 2)),
        "upper": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_two_sided": float(min(1.0, p_two)),
    }


def youden_threshold(y_true, y_score) -> float:
    y, p = _clean(y_true, y_score)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps, ys = p[order], y[order]
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ps))[0], len(ps) - 1]
    j = tps[last] / ys.sum() - fps[last] / (len(ys) - ys.sum())
    return float(ps[last][int(np.argmax(j))])


def results_frame(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T
