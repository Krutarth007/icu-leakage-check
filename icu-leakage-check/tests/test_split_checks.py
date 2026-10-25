import numpy as np
import pandas as pd

from icu_leakage_check import audit, group_split, site_split, temporal_split
from icu_leakage_check.findings import Status


def _status(result, check_id, leakage_type=None):
    fs = [f for f in result.by_check(check_id) if leakage_type is None or f.leakage_type == leakage_type]
    return {f.status for f in fs}


def test_row_split_with_repeat_patients_fails(cohort):
    rng = np.random.default_rng(0)
    cohort["split"] = np.where(rng.random(len(cohort)) < 0.2, "test", "train")
    r = audit(cohort, "died", "split", dataset="eicu", checks=["entity_overlap"])
    assert Status.FAIL in _status(r, "entity_overlap")
    f = [x for x in r.by_check("entity_overlap") if x.status == Status.FAIL][0]
    assert f.evidence["level"] == "patient"
    assert f.evidence["shared_ids"] > 0


def test_patient_level_split_passes(cohort):
    cohort["split"] = group_split(cohort, "uniquepid", test_size=0.25, stratify="died")
    r = audit(cohort, "died", "split", dataset="eicu", checks=["entity_overlap"])
    assert _status(r, "entity_overlap") == {Status.PASS}
    assert set(cohort["split"]) == {"train", "test"}


def test_group_split_with_validation_has_no_overlap(cohort):
    s = group_split(cohort, "uniquepid", test_size=0.2, valid_size=0.2, stratify="died")
    parts = {k: set(cohort.loc[s == k, "uniquepid"]) for k in ("train", "valid", "test")}
    assert not (parts["train"] & parts["valid"]) and not (parts["train"] & parts["test"]) and not (parts["valid"] & parts["test"])


def test_site_overlap_only_fails_when_site_split_claimed(cohort):
    cohort["split"] = group_split(cohort, "uniquepid", test_size=0.3)
    r1 = audit(cohort, "died", "split", dataset="eicu", split_strategy="site", checks=["site_overlap"])
    assert Status.FAIL in _status(r1, "site_overlap")
    r2 = audit(cohort, "died", "split", dataset="eicu", checks=["site_overlap"])
    assert _status(r2, "site_overlap") == {Status.PASS}
    cohort["split"] = site_split(cohort, "hospitalid", test_sites=[1, 2, 3])
    r3 = audit(cohort, "died", "split", dataset="eicu", split_strategy="site", checks=["site_overlap"])
    assert _status(r3, "site_overlap") == {Status.PASS}


def test_duplicate_records_detected(cohort):
    cohort["split"] = group_split(cohort, "uniquepid", test_size=0.2)
    train = cohort[cohort.split == "train"]
    dup = train.sample(60, random_state=0).copy()
    dup["split"] = "test"
    dup["uniquepid"] = dup["uniquepid"] + "_copy"
    dup["patientunitstayid"] = dup["patientunitstayid"] + 10_000_000
    df = pd.concat([cohort, dup], ignore_index=True)
    feats = ["age", "gcs", "vent", "lactate", "heartrate"]
    r = audit(df, "died", "split", dataset="eicu", feature_columns=feats, checks=["duplicate_records"])
    assert Status.FAIL in _status(r, "duplicate_records")
    r2 = audit(cohort, "died", "split", dataset="eicu", feature_columns=feats, checks=["duplicate_records"])
    assert _status(r2, "duplicate_records") == {Status.PASS}


def test_temporal_order_in_mimic_requires_calendar_variable():
    df = pd.DataFrame({
        "subject_id": range(10), "hadm_id": range(100, 110), "y": [0, 1] * 5,
        "admittime": pd.date_range("2130-01-01", periods=10, freq="365D"),
        "split": ["train"] * 7 + ["test"] * 3, "x": np.arange(10.0),
    })
    r = audit(df, "y", "split", dataset="mimic-iv", index_time="admittime", split_strategy="temporal", checks=["temporal_order"])
    assert Status.FAIL in _status(r, "temporal_order")
    df["anchor_year_group"] = ["2008 - 2010"] * 4 + ["2011 - 2013"] * 3 + ["2017 - 2019"] * 3
    r2 = audit(df, "y", "split", dataset="mimic-iv", calendar_time="anchor_year_group", split_strategy="temporal",
               exclude_columns=["admittime"], checks=["temporal_order"])
    assert _status(r2, "temporal_order") == {Status.PASS}


def test_temporal_split_and_violation(cohort):
    cohort["split"] = temporal_split(cohort, "hospitaldischargeyear", test_from=2015)
    r = audit(cohort, "died", "split", dataset="eicu", calendar_time="hospitaldischargeyear", split_strategy="temporal", checks=["temporal_order"])
    assert _status(r, "temporal_order") == {Status.PASS}
    cohort.loc[cohort.index[:20], "split"] = "test"
    cohort.loc[cohort.index[:20], "hospitaldischargeyear"] = 2014
    r2 = audit(cohort, "died", "split", dataset="eicu", calendar_time="hospitaldischargeyear", split_strategy="temporal", checks=["temporal_order"])
    assert Status.FAIL in _status(r2, "temporal_order")


def test_temporal_split_parses_year_groups():
    df = pd.DataFrame({"g": ["2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019"]})
    s = temporal_split(df, "g", test_from="2017 - 2019", valid_from="2014 - 2016")
    assert s.tolist() == ["train", "train", "valid", "test"]
