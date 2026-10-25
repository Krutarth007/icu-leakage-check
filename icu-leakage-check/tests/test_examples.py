"""Run the example scripts end to end on small simulated databases in the nested-folder layout."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from .conftest import write_nested

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_eicu(root: Path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(1, n + 1)
    died = rng.random(n) < 0.15
    unit_los = rng.uniform(1500, 9000, n)
    patient = pd.DataFrame({
        "patientunitstayid": ids, "patienthealthsystemstayid": ids + 10000, "uniquepid": [f"U{i // 2}" for i in ids],
        "hospitalid": rng.integers(1, 15, n), "age": np.where(rng.random(n) < 0.05, "> 89", rng.integers(20, 89, n).astype(str)),
        "gender": rng.choice(["Male", "Female"], n), "unitvisitnumber": 1,
        "hospitaldischargestatus": np.where(died, "Expired", "Alive"), "hospitaldischargeoffset": unit_los + 500,
        "unitdischargeoffset": unit_los, "hospitaldischargeyear": rng.choice([2014, 2015], n),
    })
    write_nested(root, None, "patient", patient)
    labs = []
    for i, d, los in zip(ids, died, unit_los):
        for name in ("lactate", "creatinine", "BUN", "potassium"):
            for t in rng.uniform(-60, los, 5):
                labs.append((i, name, rng.normal(2 + 1.5 * d * (t > 1440), 0.5), t, t + rng.uniform(10, 200)))
    lab = pd.DataFrame(labs, columns=["patientunitstayid", "labname", "labresult", "labresultoffset", "labresultrevisedoffset"])
    lab.insert(0, "labid", range(len(lab)))
    write_nested(root, None, "lab", lab)
    va = pd.DataFrame({"vitalaperiodicid": range(n * 4), "patientunitstayid": np.repeat(ids, 4),
                       "observationoffset": rng.uniform(0, 3000, n * 4), "noninvasivemean": rng.normal(75, 10, n * 4),
                       "noninvasivesystolic": rng.normal(115, 15, n * 4)})
    write_nested(root, None, "vitalAperiodic", va)
    vp = pd.DataFrame({"vitalperiodicid": range(n * 6), "patientunitstayid": np.repeat(ids, 6),
                       "observationoffset": rng.uniform(0, 3000, n * 6), "heartrate": rng.normal(90, 12, n * 6),
                       "respiration": rng.normal(18, 4, n * 6), "sao2": rng.normal(96, 2, n * 6)})
    write_nested(root, None, "vitalPeriodic", vp)


def _fake_mimic(root: Path, n=300, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2150-01-01")
    subj = np.arange(1, n + 1)
    hadm = subj + 20000
    stay = subj + 30000
    admit = t0 + pd.to_timedelta(rng.uniform(0, 3000, n), unit="D")
    intime = admit + pd.to_timedelta(rng.uniform(1, 20, n), unit="h")
    outtime = intime + pd.to_timedelta(rng.uniform(26, 200, n), unit="h")
    died = rng.random(n) < 0.15
    disch = outtime + pd.to_timedelta(rng.uniform(1, 100, n), unit="h")
    write_nested(root, "icu", "icustays", pd.DataFrame({
        "subject_id": subj, "hadm_id": hadm, "stay_id": stay, "first_careunit": "MICU", "last_careunit": "MICU",
        "intime": intime, "outtime": outtime, "los": (outtime - intime).total_seconds() / 86400}), gz=True)
    write_nested(root, "hosp", "admissions", pd.DataFrame({
        "subject_id": subj, "hadm_id": hadm, "admittime": admit, "dischtime": disch,
        "deathtime": np.where(died, disch.astype(str), ""), "hospital_expire_flag": died.astype(int)}), gz=True)
    write_nested(root, "hosp", "patients", pd.DataFrame({
        "subject_id": subj, "gender": rng.choice(["M", "F"], n), "anchor_age": rng.integers(20, 90, n),
        "anchor_year": admit.year, "anchor_year_group": rng.choice(["2008 - 2010", "2017 - 2019"], n)}))
    labs, charts = [], []
    for s, h, it, ot, d in zip(stay, hadm, intime, outtime, died):
        for item in (50813, 50912, 50971):
            for _ in range(5):
                ct = it + pd.Timedelta(hours=float(rng.uniform(-4, (ot - it).total_seconds() / 3600)))
                labs.append((h, item, ct, ct + pd.Timedelta(minutes=float(rng.uniform(20, 180))), rng.normal(2 + d, 0.5)))
        for item in (220045, 220210):
            for _ in range(6):
                ct = it + pd.Timedelta(hours=float(rng.uniform(0, (ot - it).total_seconds() / 3600)))
                charts.append((s, item, ct, ct + pd.Timedelta(minutes=5), rng.normal(85, 10)))
    write_nested(root, "hosp", "labevents", pd.DataFrame(labs, columns=["hadm_id", "itemid", "charttime", "storetime", "valuenum"]), gz=True)
    write_nested(root, "icu", "chartevents", pd.DataFrame(charts, columns=["stay_id", "itemid", "charttime", "storetime", "valuenum"]))
    dx = pd.DataFrame({"subject_id": np.repeat(subj, 2), "hadm_id": np.repeat(hadm, 2), "seq_num": np.tile([1, 2], n),
                       "icd_code": np.where(np.repeat(died, 2) & (np.tile([1, 2], n) == 2), "Z66", "I10"), "icd_version": 10})
    write_nested(root, "hosp", "diagnoses_icd", dx)


@pytest.mark.parametrize("vital_periodic", [True, False])
def test_eicu_example(tmp_path, vital_periodic):
    _fake_eicu(tmp_path / "eicu")
    mod = _load("eicu_mortality_24h")
    res = mod.run(tmp_path / "eicu", tmp_path / "out", use_vital_periodic=vital_periodic, n_boot=50)
    assert not res["clean"].has_problems("high"), res["clean"].summary()
    assert res["leaky"].has_problems("critical")
    assert "unitdischargeoffset" in res["leaky"].leaky_features("high")
    assert (tmp_path / "out" / "eicu_leaky_report.html").exists()


def test_mimic_example(tmp_path):
    _fake_mimic(tmp_path / "mimic")
    mod = _load("mimic_iv_mortality_24h")
    res = mod.run(tmp_path / "mimic", tmp_path / "out", n_boot=50)
    assert not res["clean"].has_problems("high"), res["clean"].summary()
    leaky = res["leaky"]
    assert leaky.has_problems("critical")
    flagged = leaky.leaky_features("high")
    assert {"los", "dx_count", "dx_proxy_code"} <= set(flagged)


def test_examples_without_duckdb(tmp_path):
    _fake_eicu(tmp_path / "eicu")
    mod = _load("eicu_mortality_24h")
    mod._has_duckdb = lambda: False
    res = mod.run(tmp_path / "eicu", tmp_path / "out_e", n_boot=30)
    assert res["leaky"].has_problems("critical") and not res["clean"].has_problems("high")
    _fake_mimic(tmp_path / "mimic")
    mm = _load("mimic_iv_mortality_24h")
    mm._has_duckdb = lambda: False
    res = mm.run(tmp_path / "mimic", tmp_path / "out_m", n_boot=30)
    assert res["leaky"].has_problems("critical") and not res["clean"].has_problems("high")
