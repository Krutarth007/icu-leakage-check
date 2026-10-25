import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from icu_leakage_check import leakage_impact
from icu_leakage_check.compare import inflation_summary, leakage_impact_table
from icu_leakage_check.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    brier,
    calibration_slope_intercept,
    expected_calibration_error,
    youden_threshold,
)


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.5 + 1.2 * x)))
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_metrics_match_sklearn():
    y, p = _data()
    p_tied = np.round(p, 2)
    assert np.isclose(auroc(y, p_tied), roc_auc_score(y, p_tied))
    assert np.isclose(auprc(y, p), average_precision_score(y, p))
    assert np.isclose(brier(y, p), brier_score_loss(y, p))


def test_calibration_of_true_probabilities():
    y, p = _data(20000)
    c = calibration_slope_intercept(y, p)
    assert abs(c["slope"] - 1) < 0.08 and abs(c["intercept"]) < 0.08
    over = calibration_slope_intercept(y, np.clip(p * 1.6, 0, 0.999))
    assert over["intercept"] < -0.2
    assert expected_calibration_error(y, p) < 0.03


def test_bootstrap_and_youden():
    y, p = _data()
    ci = bootstrap_ci(auroc, y, p, n_boot=200)
    assert ci["lower"] < ci["estimate"] < ci["upper"]
    groups = np.repeat(np.arange(len(y) // 3), 3)
    ci_g = bootstrap_ci(auroc, y, p, n_boot=100, groups=groups)
    assert ci_g["lower"] < ci_g["upper"]
    t = youden_threshold(y, p)
    assert 0 < t < 1


def test_leakage_impact_detects_inflation():
    y, p = _data()
    rng = np.random.default_rng(1)
    leaky = np.clip(p + 0.35 * y - 0.1 + rng.normal(0, 0.02, len(y)), 0.001, 0.999)
    df = leakage_impact(y, p, leaky, n_boot=200)
    row = df.set_index("metric").loc["auroc"]
    assert row["difference"] > 0.05 and row["diff_lower"] > 0
    table = leakage_impact_table(y, p, {"leak_a": leaky, "none": p}, n_boot=100)
    summ = inflation_summary(table)
    assert summ.iloc[0]["variant"] == "leak_a" and bool(summ.iloc[0]["significant"])
