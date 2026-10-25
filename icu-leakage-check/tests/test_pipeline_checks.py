import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from icu_leakage_check import audit, group_split
from icu_leakage_check.findings import Status

FEATS = ["age", "gcs", "vent", "lactate", "heartrate"]


@pytest.fixture
def split_cohort(cohort):
    cohort["split"] = group_split(cohort, "uniquepid", test_size=0.3, stratify="died")
    cohort.loc[cohort.sample(frac=0.1, random_state=1).index, "lactate"] = np.nan
    return cohort


def test_scaler_fitted_on_all_rows_is_detected(split_cohort):
    sc = StandardScaler().fit(split_cohort[FEATS])
    r = audit(split_cohort, "died", "split", dataset="eicu", preprocessor=sc, preprocessor_columns=FEATS, checks=["preprocessing_fit"])
    assert r.by_check("preprocessing_fit")[0].status == Status.FAIL


def test_scaler_fitted_on_train_passes(split_cohort):
    tr = split_cohort[split_cohort.split == "train"]
    sc = StandardScaler().fit(tr[FEATS])
    r = audit(split_cohort, "died", "split", dataset="eicu", preprocessor=sc, preprocessor_columns=FEATS, checks=["preprocessing_fit"])
    assert r.by_check("preprocessing_fit")[0].status == Status.PASS


def test_nested_pipeline_inspection(split_cohort):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), ["age", "lactate", "heartrate"]),
        ("mm", MinMaxScaler(), ["gcs"]),
    ])
    model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
    model.fit(split_cohort[FEATS], split_cohort["died"])  # fitted on everything: leaky
    r = audit(split_cohort, "died", "split", dataset="eicu", preprocessor=model, preprocessor_columns=FEATS, checks=["preprocessing_fit"])
    f = r.by_check("preprocessing_fit")[0]
    assert f.status == Status.FAIL
    assert any("impute.statistics_" in a for a in f.affected)

    tr = split_cohort[split_cohort.split == "train"]
    model.fit(tr[FEATS], tr["died"])
    r2 = audit(split_cohort, "died", "split", dataset="eicu", preprocessor=model, preprocessor_columns=FEATS, checks=["preprocessing_fit"])
    assert r2.by_check("preprocessing_fit")[0].status == Status.PASS


def _smote(X, y, seed=0):
    from imblearn.over_sampling import SMOTE

    return SMOTE(random_state=seed).fit_resample(X, y)


def test_smote_before_split_is_detected(cohort):
    pytest.importorskip("imblearn")
    X, y = _smote(cohort[["age", "gcs", "vent", "heartrate", "lactate"]], cohort["died"])
    df = X.assign(died=y.values)
    rng = np.random.default_rng(0)
    df["split"] = np.where(rng.random(len(df)) < 0.3, "test", "train")
    r = audit(df, "died", "split", checks=["synthetic_samples"])
    assert r.by_check("synthetic_samples")[0].status == Status.FAIL


def test_smote_inside_training_passes(cohort):
    pytest.importorskip("imblearn")
    cohort["split"] = group_split(cohort, "uniquepid", test_size=0.3, stratify="died")
    feats = ["age", "gcs", "vent", "heartrate", "lactate"]
    tr = cohort[cohort.split == "train"]
    X, y = _smote(tr[feats], tr["died"])
    train_res = X.assign(died=y.values, split="train")
    test = cohort.loc[cohort.split == "test", feats + ["died", "split"]]
    df = pd.concat([train_res, test], ignore_index=True)
    r = audit(df, "died", "split", checks=["synthetic_samples"])
    f = r.by_check("synthetic_samples")[0]
    assert f.status == Status.PASS and "confined to the training split" in f.message


def test_threshold_tuned_on_test(split_cohort):
    from icu_leakage_check.metrics import youden_threshold

    rng = np.random.default_rng(0)
    split_cohort["score"] = np.clip(split_cohort["died"] * 0.3 + rng.uniform(0, 0.7, len(split_cohort)), 0, 1)
    te = split_cohort[split_cohort.split == "test"]
    t = youden_threshold(te["died"], te["score"])
    r = audit(split_cohort, "died", "split", dataset="eicu", scores="score", threshold=t, checks=["threshold_selection"])
    assert r.by_check("threshold_selection")[0].status == Status.WARN
    r2 = audit(split_cohort, "died", "split", dataset="eicu", scores="score", threshold=0.5, checks=["threshold_selection"])
    assert r2.by_check("threshold_selection")[0].status == Status.PASS


def test_declarations(cohort):
    r = audit(cohort, "died", dataset="eicu", declarations={"feature_selection_on_training_only": False, "test_set_used_once": True}, checks=["declarations"])
    statuses = {f.status for f in r.by_check("declarations")}
    assert {Status.WARN, Status.PASS, Status.SKIPPED} <= statuses
