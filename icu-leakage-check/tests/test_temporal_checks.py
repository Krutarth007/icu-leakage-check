import numpy as np
import pandas as pd

from icu_leakage_check import audit, build_window_features, make_feature_builder
from icu_leakage_check.findings import Status


def _by_type(result, check_id):
    return {f.leakage_type: f for f in result.by_check(check_id)}


def test_event_timing_flags_each_mechanism():
    data = pd.DataFrame({"stay": [1, 2, 3], "y": [0, 1, 0], "t": [100.0, 100.0, 100.0],
                         "lab__max": [1.0, 2.0, 3.0], "drug__sum": [1.0, 1.0, 1.0], "dx__count": [1, 0, 1]})
    events = pd.DataFrame([
        {"entity_id": 1, "variable": "lab", "event_time": 50.0, "available_time": 60.0, "end_time": np.nan},
        {"entity_id": 2, "variable": "lab", "event_time": 300.0, "available_time": 310.0, "end_time": np.nan},
        {"entity_id": 3, "variable": "lab", "event_time": 90.0, "available_time": 400.0, "end_time": np.nan},
        {"entity_id": 1, "variable": "drug", "event_time": 80.0, "available_time": 80.0, "end_time": 500.0},
        {"entity_id": 2, "variable": "dx", "event_time": np.nan, "available_time": np.nan, "end_time": np.nan},
    ])
    r = audit(data, "y", stay_id="stay", index_time="t", events=events, checks=["event_timing"])
    f = _by_type(r, "event_timing")
    assert f["FUTURE_EVENT"].status == Status.FAIL and "lab__max" in f["FUTURE_EVENT"].affected
    assert f["AVAILABILITY_LAG"].status == Status.FAIL
    assert f["INTERVAL_CROSSING"].status == Status.FAIL and "drug__sum" in f["INTERVAL_CROSSING"].affected
    assert f["UNDATED_SOURCE"].status == Status.FAIL and "dx__count" in f["UNDATED_SOURCE"].affected
    assert r.has_problems("critical")


def test_safe_feature_builder_provenance_passes(cohort, events_for):
    idx = cohort[["patientunitstayid", "index_offset"]].rename(columns={"patientunitstayid": "entity_id", "index_offset": "index_time"})
    feats, prov = build_window_features(events_for, idx, window=1440, return_provenance=True)
    df = cohort.merge(feats, left_on="patientunitstayid", right_index=True, how="left")
    r = audit(df, "died", dataset="eicu", index_time="index_offset", events=prov,
              feature_columns=list(feats.columns), checks=["event_timing"])
    assert all(f.status == Status.PASS for f in r.by_check("event_timing"))


def test_leaky_features_fail_event_timing(cohort, events_for):
    idx = cohort[["patientunitstayid", "index_offset"]].rename(columns={"patientunitstayid": "entity_id", "index_offset": "index_time"})
    feats, prov = build_window_features(events_for, idx, cutoff="none", return_provenance=True)
    df = cohort.merge(feats, left_on="patientunitstayid", right_index=True, how="left")
    r = audit(df, "died", dataset="eicu", index_time="index_offset", events=prov,
              feature_columns=list(feats.columns), checks=["event_timing"])
    f = _by_type(r, "event_timing")["FUTURE_EVENT"]
    assert f.status == Status.FAIL and "creatinine__max" in f.affected


def test_truncation_invariance_detects_whole_stay_features(cohort, events_for):
    def leaky_builder(events, index_times):
        return events.groupby("entity_id")["value"].agg(["max", "mean"]).add_prefix("creat_")

    r = audit(cohort, "died", dataset="eicu", index_time="index_offset", raw_events=events_for,
              feature_builder=leaky_builder, checks=["truncation_invariance"])
    assert r.by_check("truncation_invariance")[0].status == Status.FAIL

    r2 = audit(cohort, "died", dataset="eicu", index_time="index_offset", raw_events=events_for,
               feature_builder=make_feature_builder(window=1440), checks=["truncation_invariance"])
    assert r2.by_check("truncation_invariance")[0].status == Status.PASS


def test_outcome_timing_and_eligibility(cohort):
    cohort.loc[cohort.died == 1, "death_offset"] = cohort.loc[cohort.died == 1, "death_offset"].values
    cohort.loc[cohort.index[cohort.died == 1][:15], "death_offset"] = 600.0
    r = audit(cohort, "died", dataset="eicu", index_time="index_offset", outcome_time="death_offset", checks=["outcome_timing"])
    f = r.by_check("outcome_timing")[0]
    assert f.status == Status.FAIL and f.evidence["outcome_at_or_before_prediction"] == 15

    cohort["eligible_at"] = 0.0
    cohort.loc[cohort.index[:30], "eligible_at"] = 2880.0
    r2 = audit(cohort, "died", dataset="eicu", index_time="index_offset", eligibility_time="eligible_at", checks=["eligibility_timing"])
    assert r2.by_check("eligibility_timing")[0].status == Status.FAIL


def test_interval_prorating():
    events = pd.DataFrame([{"entity_id": 1, "variable": "norepi", "value": 100.0, "event_time": 0.0,
                            "available_time": 0.0, "end_time": 200.0}])
    idx = pd.DataFrame({"entity_id": [1], "index_time": [50.0]})
    f = build_window_features(events, idx, aggregations=("sum",))
    assert np.isclose(f.loc[1, "norepi__sum"], 25.0)
    f2 = build_window_features(events, idx, aggregations=("sum",), interval_policy="keep")
    assert np.isclose(f2.loc[1, "norepi__sum"], 100.0)
    f3 = build_window_features(events, idx, aggregations=("sum",), interval_policy="exclude")
    assert "norepi__sum" not in f3.columns or np.isnan(f3.loc[1, "norepi__sum"])


def test_availability_basis_excludes_late_results():
    events = pd.DataFrame([
        {"entity_id": 1, "variable": "lac", "value": 2.0, "event_time": 10.0, "available_time": 20.0, "end_time": np.nan},
        {"entity_id": 1, "variable": "lac", "value": 9.0, "event_time": 90.0, "available_time": 150.0, "end_time": np.nan},
    ])
    idx = pd.DataFrame({"entity_id": [1], "index_time": [100.0]})
    safe = build_window_features(events, idx, aggregations=("last",))
    leaky = build_window_features(events, idx, aggregations=("last",), time_basis="event")
    assert safe.loc[1, "lac__last"] == 2.0 and leaky.loc[1, "lac__last"] == 9.0


def test_datetime_events_work():
    t0 = pd.Timestamp("2150-01-01 08:00")
    events = pd.DataFrame([
        {"entity_id": 7, "variable": "k", "value": 4.0, "event_time": t0, "available_time": t0 + pd.Timedelta(hours=1), "end_time": pd.NaT},
        {"entity_id": 7, "variable": "k", "value": 6.0, "event_time": t0 + pd.Timedelta(hours=30), "available_time": t0 + pd.Timedelta(hours=31), "end_time": pd.NaT},
    ])
    idx = pd.DataFrame({"entity_id": [7], "index_time": [t0 + pd.Timedelta(hours=24)]})
    f, prov = build_window_features(events, idx, window=25, aggregations=("max", "time_since_last"), return_provenance=True)
    assert f.loc[7, "k__max"] == 4.0 and np.isclose(f.loc[7, "k__time_since_last"], 23.0)
    data = pd.DataFrame({"stay_id": [7], "y": [1], "t": [t0 + pd.Timedelta(hours=24)], "k__max": [6.0]})
    r = audit(data, "y", stay_id="stay_id", index_time="t", events=events, checks=["event_timing"])
    assert {f.leakage_type: f.status for f in r.by_check("event_timing")}["FUTURE_EVENT"] == Status.FAIL
