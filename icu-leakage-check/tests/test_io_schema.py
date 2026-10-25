import numpy as np
import pandas as pd
import pytest

from icu_leakage_check import (
    EICU,
    MIMIC_IV,
    attach_entity,
    convert_to_parquet,
    events_from_table,
    find_table,
    index_times_from_start,
    list_tables,
    read_table,
)
from icu_leakage_check.schemas.base import UNTIMED

from .conftest import write_nested

MIMIC_HOSP = ["admissions", "d_hcpcs", "d_icd_diagnoses", "d_icd_procedures", "d_labitems", "diagnoses_icd", "drgcodes",
              "emar", "emar_detail", "hcpcsevents", "labevents", "microbiologyevents", "omr", "patients", "pharmacy", "poe",
              "poe_detail", "prescriptions", "procedures_icd", "provider", "services", "transfers"]
MIMIC_ICU = ["caregiver", "chartevents", "d_items", "datetimeevents", "icustays", "ingredientevents", "inputevents",
             "outputevents", "procedureevents"]
EICU_TABLES = ["vitalPeriodic", "vitalAperiodic", "treatment", "respiratoryCare", "respiratoryCharting", "patient",
               "physicalExam", "pastHistory", "nurseCharting", "nurseCare", "nurseAssessment", "microLab", "note",
               "medication", "lab", "intakeOutput", "carePlanInfectiousDisease", "customLab", "diagnosis", "hospital",
               "infusionDrug", "apachePredVar", "carePlanCareProvider", "carePlanEOL", "carePlanGeneral", "carePlanGoal",
               "admissionDrug", "admissionDx", "allergy", "apacheApsVar", "apachePatientResult"]


def test_registries_cover_every_distributed_table():
    for t in MIMIC_HOSP:
        assert MIMIC_IV.table(t).module == "hosp"
    for t in MIMIC_ICU:
        assert MIMIC_IV.table(t).module == "icu"
    for t in EICU_TABLES:
        assert EICU.has_table(t)
    assert len(MIMIC_IV.tables) == len(MIMIC_HOSP) + len(MIMIC_ICU)
    assert len(EICU.tables) == len(EICU_TABLES)
    assert MIMIC_IV.table("diagnoses_icd").timing == UNTIMED
    assert "diedinhospital" in EICU.table("apachePredVar").outcome_columns


def test_nested_layout_and_filters(tmp_path):
    lab = pd.DataFrame({
        "labid": range(6), "patientUnitStayID": [1, 1, 2, 2, 3, 3], "labResultOffset": [10, 50, 20, 900, 5, 3000],
        "labname": ["lactate"] * 6, "labresult": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "labresultrevisedoffset": [30, np.nan, 25, 910, 6, 3005],
    })
    write_nested(tmp_path, None, "lab", lab)
    p = find_table(tmp_path, "lab", "eicu")
    assert p.name == "lab.csv" and p.parent.name == "lab.csv"
    df = read_table(tmp_path, "lab", "eicu", filters={"patientunitstayid": [1, 2]}, engine="pandas")
    assert set(df["patientunitstayid"]) == {1, 2} and "labresultoffset" in df.columns
    df2 = read_table(tmp_path, "lab", "eicu", columns=["labname", "labresult"], filters={"patientunitstayid": [3]}, engine="duckdb")
    assert len(df2) == 2 and set(df2.columns) == {"labname", "labresult", "patientunitstayid"}


def test_mimic_layout_gz_and_list(tmp_path):
    icu = pd.DataFrame({"subject_id": [1, 2], "hadm_id": [10, 20], "stay_id": [100, 200],
                        "first_careunit": ["MICU", "SICU"], "last_careunit": ["MICU", "SICU"],
                        "intime": ["2150-01-01 08:00:00", "2160-05-05 10:00:00"], "outtime": ["2150-01-03 08:00:00", "2160-05-06 10:00:00"],
                        "los": [2.0, 1.0]})
    write_nested(tmp_path, "icu", "icustays", icu, gz=True)
    write_nested(tmp_path, "hosp", "patients", pd.DataFrame({"subject_id": [1, 2], "gender": ["F", "M"], "anchor_age": [60, 70],
                                                               "anchor_year": [2150, 2160], "anchor_year_group": ["2008 - 2010", "2017 - 2019"], "dod": [None, None]}))
    df = read_table(tmp_path, "icustays", "mimic-iv")
    assert pd.api.types.is_datetime64_any_dtype(df["intime"])
    listing = list_tables(tmp_path, "mimic-iv")
    assert listing.loc[listing.table == "icustays", "found"].item()
    assert listing.loc[listing.table == "patients", "found"].item()
    assert not listing.loc[listing.table == "chartevents", "found"].item()


def test_convert_to_parquet(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    vp = pd.DataFrame({"vitalperiodicid": [1, 2], "patientunitstayid": [5, 5], "observationOffset": [5, 10], "heartrate": [80, 85], "sao2": [97, 96]})
    write_nested(tmp_path / "raw", None, "vitalPeriodic", vp)
    out = convert_to_parquet(tmp_path / "raw", "eicu", tmp_path / "pq", tables=["vitalPeriodic", "lab"], verbose=False)
    assert out.set_index("table").loc["vitalPeriodic", "status"] == "converted"
    assert out.set_index("table").loc["lab", "status"] == "missing"
    df = read_table(tmp_path / "pq", "vitalPeriodic", "eicu")
    assert "observationoffset" in df.columns and len(df) == 2


def test_eicu_events_wide_and_revised():
    vp = pd.DataFrame({"vitalperiodicid": [1, 2], "patientunitstayid": [5, 5], "observationoffset": [5, 10],
                       "heartrate": [80, np.nan], "sao2": [97, 96]})
    ev = events_from_table(vp, "vitalPeriodic", "eicu")
    assert set(ev["variable"]) == {"heartrate", "sao2"} and len(ev) == 3
    lab = pd.DataFrame({"labid": [1, 2], "patientunitstayid": [5, 5], "labresultoffset": [100, 200], "labname": ["k", "k"],
                        "labresult": [4.0, 5.0], "labresultrevisedoffset": [150, np.nan]})
    le = events_from_table(lab, "lab", "eicu")
    assert le["available_time"].tolist() == [150, 200]
    aps = pd.DataFrame({"apacheapsvarid": [1], "patientunitstayid": [5], "creatinine": [1.1], "heartrate": [120]})
    fe = events_from_table(aps, "apacheApsVar", "eicu")
    assert (fe["available_time"] == 1440).all()


def test_mimic_events_storetime_dates_and_untimed():
    labs = pd.DataFrame({"labevent_id": [1], "subject_id": [1], "hadm_id": [10], "specimen_id": [3], "itemid": [50813],
                         "charttime": pd.to_datetime(["2150-01-01 08:00"]), "storetime": pd.to_datetime(["2150-01-01 09:30"]), "valuenum": [2.5]})
    ev = events_from_table(labs, "labevents", "mimic-iv")
    assert ev.loc[0, "available_time"] == pd.Timestamp("2150-01-01 09:30") and ev.loc[0, "variable"] == "50813"
    proc = pd.DataFrame({"subject_id": [1], "hadm_id": [10], "seq_num": [1], "chartdate": pd.to_datetime(["2150-01-02"]), "icd_code": ["5A1955Z"], "icd_version": [10]})
    pe = events_from_table(proc, "procedures_icd", "mimic-iv")
    assert pe.loc[0, "available_time"] == pd.Timestamp("2150-01-03")
    dx = pd.DataFrame({"subject_id": [1], "hadm_id": [10], "seq_num": [1], "icd_code": ["Z66"], "icd_version": [10]})
    de = events_from_table(dx, "diagnoses_icd", "mimic-iv", encounter_end={10: pd.Timestamp("2150-01-09")})
    assert pd.isna(de.loc[0, "event_time"]) and de.loc[0, "available_time"] == pd.Timestamp("2150-01-09") and bool(de.loc[0, "hindsight"])


def test_attach_entity_and_index_times():
    ev = pd.DataFrame({"entity_id": [10, 10], "variable": ["k", "k"], "value": [1.0, 2.0],
                       "event_time": pd.to_datetime(["2150-01-01 09:00", "2150-01-05 09:00"]),
                       "available_time": pd.NaT, "end_time": pd.NaT})
    stays = pd.DataFrame({"hadm_id": [10], "stay_id": [100], "intime": pd.to_datetime(["2150-01-01 08:00"]), "outtime": pd.to_datetime(["2150-01-03 08:00"])})
    out = attach_entity(ev, stays, "hadm_id", "stay_id", "intime", "outtime")
    assert out["entity_id"].tolist() == [100]
    idx = index_times_from_start(stays, "stay_id", "intime", hours=24)
    assert idx.loc[0, "index_time"] == pd.Timestamp("2150-01-02 08:00")
    idx2 = index_times_from_start(pd.DataFrame({"patientunitstayid": [1, 2]}), "patientunitstayid", hours=6)
    assert (idx2["index_time"] == 360).all()


def test_where_row_filter_and_query(tmp_path):
    pytest.importorskip("duckdb")
    from icu_leakage_check import query_table

    vp = pd.DataFrame({"vitalperiodicid": range(6), "patientunitstayid": [1, 1, 1, 2, 2, 2],
                       "observationoffset": [5, 65, 3000, 10, 70, 80], "heartrate": [80, 90, 100, 60, 62, 64]})
    write_nested(tmp_path, None, "vitalPeriodic", vp)
    a = read_table(tmp_path, "vitalPeriodic", "eicu", filters={"patientunitstayid": [1]}, where="observationoffset <= 2880")
    assert a["observationoffset"].tolist() == [5, 65]
    b = read_table(tmp_path, "vitalPeriodic", "eicu", engine="pandas", row_filter=lambda d: d[d.heartrate > 85])
    assert b["heartrate"].tolist() == [90, 100]
    q = query_table(tmp_path, "vitalPeriodic",
                    "SELECT patientunitstayid, avg(heartrate) AS hr FROM {table} WHERE patientunitstayid IN (SELECT v FROM ids) GROUP BY 1",
                    dataset="eicu", params={"ids": [2]})
    assert q["hr"].tolist() == [62.0]
