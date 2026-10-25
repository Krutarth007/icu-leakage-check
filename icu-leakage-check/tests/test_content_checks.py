import numpy as np
import pandas as pd

from icu_leakage_check import audit, find_proxy_items, flag_proxy_codes
from icu_leakage_check.findings import Severity, Status


def _types(result, check_id):
    return {f.leakage_type: f for f in result.by_check(check_id)}


def test_eicu_registry_flags_outcome_hindsight_and_first_day(cohort):
    cohort["diedinhospital"] = cohort["died"]
    cohort["unitdischargeoffset"] = 3000
    cohort["apachescore"] = 50
    cohort["index_offset"] = 360.0
    r = audit(cohort, "died", dataset="eicu", index_time="index_offset", checks=["hindsight_features"])
    t = _types(r, "hindsight_features")
    assert "diedinhospital" in t["OUTCOME_ENCODING"].affected and t["OUTCOME_ENCODING"].severity == Severity.CRITICAL
    assert "unitdischargeoffset" in t["HINDSIGHT_VARIABLE"].affected
    assert "apachescore" in t["FIRST_DAY_SUMMARY"].affected and t["FIRST_DAY_SUMMARY"].status == Status.FAIL
    for clean in ("age", "gcs", "lactate"):
        assert all(clean not in f.affected for f in r.findings)


def test_first_day_ok_after_24_hours(cohort):
    cohort["apachescore"] = 50
    r = audit(cohort, "died", dataset="eicu", index_time="index_offset", checks=["hindsight_features"])
    assert "FIRST_DAY_SUMMARY" not in _types(r, "hindsight_features")


def test_feature_sources_for_mimic_billing_codes():
    df = pd.DataFrame({"hadm_id": range(50), "y": [0, 1] * 25, "dx_sepsis": [0, 1] * 25,
                       "dx_chf_prior": [1, 0] * 25, "los": np.arange(50.0), "x": np.random.default_rng(0).normal(size=50)})
    sources = {"dx_sepsis": "diagnoses_icd", "dx_chf_prior": "diagnoses_icd.prior", "los": "icustays.los"}
    r = audit(df, "y", dataset="mimic-iv", feature_sources=sources, checks=["hindsight_features"])
    t = _types(r, "hindsight_features")
    assert t["UNDATED_SOURCE"].affected == ["dx_sepsis"]
    assert "los" in t["HINDSIGHT_VARIABLE"].affected
    assert all("dx_chf_prior" not in f.affected for f in r.findings)


def test_generic_names_are_warnings_not_failures():
    df = pd.DataFrame({"y": [0, 1, 0, 1], "hospital_los_days": [1, 2, 3, 4], "mortality_flag_30d": [0, 1, 0, 1], "age": [50, 60, 70, 80]})
    r = audit(df, "y", checks=["hindsight_features"])
    t = _types(r, "hindsight_features")
    assert t["HINDSIGHT_VARIABLE"].status == Status.WARN
    assert "mortality_flag_30d" in t["OUTCOME_ENCODING"].affected


def test_label_as_feature_is_critical():
    df = pd.DataFrame({"y": [0, 1, 0, 1], "a": [1, 2, 3, 4]})
    r = audit(df, "y", feature_columns=["a", "y"], checks=["hindsight_features"])
    assert _types(r, "hindsight_features")["OUTCOME_ENCODING"].severity == Severity.CRITICAL


def test_outcome_proxies():
    df = pd.DataFrame({
        "y": [0, 1] * 20, "icd10_Z66": [0, 1] * 20, "code_status": ["Full code", "DNR"] * 20,
        "careplan_item": ["Full therapy", "Comfort measures only"] * 20, "heart_rate": np.arange(40.0),
        "cardiac_arrest_dx_I469": [0, 0, 1, 0] * 10,
    })
    r = audit(df, "y", checks=["outcome_proxies"])
    f = r.by_check("outcome_proxies")[0]
    assert f.status == Status.WARN and f.severity == Severity.HIGH
    for name in ("icd10_Z66", "code_status", "careplan_item"):
        assert name in f.affected
    assert "heart_rate" not in f.affected


def test_eol_table_source_is_proxy(cohort):
    cohort["eol_discussion"] = 0
    r = audit(cohort, "died", dataset="eicu", feature_sources={"eol_discussion": "carePlanEOL"}, checks=["outcome_proxies"])
    assert "eol_discussion" in r.by_check("outcome_proxies")[0].affected


def test_flag_proxy_codes_and_items():
    out = flag_proxy_codes(["Z66", "I10", "V66.7", "427.5, 401.9", "G93.82"])
    assert set(out["meaning"]) >= {"Do not resuscitate status", "Encounter for palliative care", "Cardiac arrest", "Brain death"}
    d_items = pd.DataFrame({"itemid": [1, 2, 3], "label": ["Code Status", "Heart Rate", "Goals of care discussion"]})
    hits = find_proxy_items(d_items)
    assert hits["itemid"].tolist() == [1, 3]


def test_single_feature_dominance_and_identity(cohort):
    rng = np.random.default_rng(0)
    cohort["leak"] = cohort["died"] * 5 + rng.normal(0, 0.5, len(cohort))
    cohort["copy"] = cohort["died"]
    r = audit(cohort, "died", dataset="eicu", feature_columns=["age", "gcs", "lactate", "leak", "copy"], checks=["single_feature_dominance"])
    fs = r.by_check("single_feature_dominance")
    assert any(f.leakage_type == "OUTCOME_ENCODING" and "copy" in f.affected for f in fs)
    assert any(f.status == Status.FAIL and "leak" in f.affected for f in fs)
    clean = audit(cohort, "died", dataset="eicu", feature_columns=["age", "gcs", "lactate"], checks=["single_feature_dominance"])
    assert clean.by_check("single_feature_dominance")[0].status == Status.PASS


def test_missingness_proxy(cohort):
    cohort["discharge_weight"] = np.where(cohort["died"] == 1, np.nan, 80.0)
    r = audit(cohort, "died", dataset="eicu", feature_columns=["age", "discharge_weight"], checks=["missingness_proxy"])
    f = r.by_check("missingness_proxy")[0]
    assert f.status == Status.FAIL and "discharge_weight" in f.affected
