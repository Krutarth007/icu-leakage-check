"""Audit a 24-hour in-hospital mortality model built on eICU-CRD.

Builds the same model twice, once leakage-safe and once with common
shortcuts, audits both and writes HTML reports plus an inflation table.
Only aggregate results are written to disk.

Usage
-----
    python eicu_mortality_24h.py --root "C:/path/to/eicu-collaborative-research-database-2.0" --out eicu_audit

Large tables (vitalPeriodic, lab) are read with filters. Converting to
Parquet first (``icu-leakage-check convert``) makes repeated runs much faster;
pass the Parquet folder as ``--root`` afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from icu_leakage_check import (
    LeakageAudit,
    build_window_features,
    evaluate,
    events_from_table,
    group_split,
    leakage_impact,
    query_table,
    read_table,
)

PREDICTION_MINUTE = 1440
LOOKAHEAD_MINUTES = 2880  # events loaded beyond the prediction time, used only by the leaky variant
LAB_NAMES = [
    "lactate", "creatinine", "BUN", "WBC x 1000", "platelets x 1000", "potassium",
    "sodium", "bicarbonate", "total bilirubin", "Hgb", "glucose", "pH",
]
VITALS = ["heartrate", "respiration", "sao2"]
AGGS = ("last", "min", "max", "mean")


def parse_age(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(">", "", regex=False).str.strip(), errors="coerce")


def build_cohort(root) -> pd.DataFrame:
    p = read_table(root, "patient", "eicu", columns=[
        "patientunitstayid", "patienthealthsystemstayid", "uniquepid", "hospitalid", "age", "gender",
        "unitvisitnumber", "hospitaldischargestatus", "hospitaldischargeoffset", "unitdischargeoffset",
        "hospitaldischargeyear",
    ])
    p["age_years"] = parse_age(p["age"])
    p = p[(p["unitvisitnumber"] == 1) & (p["age_years"] >= 18) & p["hospitaldischargestatus"].isin(["Alive", "Expired"])]
    p = p[p["unitdischargeoffset"] >= PREDICTION_MINUTE]  # still in the unit at the prediction time
    p = p.copy()
    p["died"] = (p["hospitaldischargestatus"] == "Expired").astype(int)
    p["death_offset"] = np.where(p["died"] == 1, p["hospitaldischargeoffset"], np.nan)
    p = p[~((p["died"] == 1) & (p["death_offset"] <= PREDICTION_MINUTE))]  # alive at the prediction time
    p["index_offset"] = float(PREDICTION_MINUTE)
    p["eligible_offset"] = float(PREDICTION_MINUTE)  # eligibility is known at the prediction time
    p["male"] = (p["gender"] == "Male").astype(float)
    return p.reset_index(drop=True)


def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


def _read_window(root, table, columns, ids, time_col):
    """Rows of the cohort up to LOOKAHEAD_MINUTES, filtered while reading."""
    if _has_duckdb():
        return read_table(root, table, "eicu", columns=columns, filters={"patientunitstayid": ids},
                          where=f"{time_col} <= {LOOKAHEAD_MINUTES}")
    return read_table(root, table, "eicu", columns=columns, filters={"patientunitstayid": ids}, engine="pandas",
                      row_filter=lambda d: d[d[time_col] <= LOOKAHEAD_MINUTES])


def _hourly_vitals(root, ids) -> pd.DataFrame:
    """vitalPeriodic is recorded every few minutes; reduce it to hourly means.

    Each hourly mean is stamped at the end of its hour, so it only becomes
    usable once the hour is complete.
    """
    if _has_duckdb():
        means = ", ".join(f"avg({v}) AS {v}" for v in VITALS)
        sql = (
            f"SELECT patientunitstayid, floor(observationoffset / 60) * 60 + 59 AS observationoffset, {means} "
            "FROM {table} WHERE patientunitstayid IN (SELECT v FROM ids) "
            f"AND observationoffset >= 0 AND observationoffset <= {LOOKAHEAD_MINUTES} GROUP BY 1, 2"
        )
        return query_table(root, "vitalPeriodic", sql, dataset="eicu", params={"ids": ids})
    vp = read_table(root, "vitalPeriodic", "eicu", columns=["patientunitstayid", "observationoffset"] + VITALS,
                    filters={"patientunitstayid": ids}, engine="pandas",
                    row_filter=lambda d: d[(d["observationoffset"] >= 0) & (d["observationoffset"] <= LOOKAHEAD_MINUTES)])
    vp["observationoffset"] = (vp["observationoffset"] // 60) * 60 + 59
    return vp.groupby(["patientunitstayid", "observationoffset"], as_index=False)[VITALS].mean()


def load_events(root, ids, use_vital_periodic: bool = True) -> pd.DataFrame:
    lab = _read_window(root, "lab", ["patientunitstayid", "labname", "labresult", "labresultoffset", "labresultrevisedoffset"],
                       ids, "labresultoffset")
    lab = lab[lab["labname"].isin(LAB_NAMES)]
    frames = [events_from_table(lab, "lab", "eicu")]
    va = _read_window(root, "vitalAperiodic", ["patientunitstayid", "observationoffset", "noninvasivemean", "noninvasivesystolic"],
                      ids, "observationoffset")
    frames.append(events_from_table(va, "vitalAperiodic", "eicu"))
    if use_vital_periodic:
        frames.append(events_from_table(_hourly_vitals(root, ids), "vitalPeriodic", "eicu", value_columns=VITALS))
    ev = pd.concat(frames, ignore_index=True)
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    return ev.dropna(subset=["value"])


def make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000)),
    ])


def run(root, out_dir="eicu_audit", use_vital_periodic=True, max_stays=None, seed=42, n_boot=500):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort(root)
    if max_stays:
        cohort = cohort.sample(min(max_stays, len(cohort)), random_state=seed).reset_index(drop=True)
    ids = cohort["patientunitstayid"].tolist()
    events = load_events(root, ids, use_vital_periodic)
    index = cohort[["patientunitstayid", "index_offset"]].rename(columns={"patientunitstayid": "entity_id", "index_offset": "index_time"})
    base = ["age_years", "male"]

    # Leakage-safe pipeline
    feats, prov = build_window_features(events, index, window=PREDICTION_MINUTE, aggregations=AGGS, return_provenance=True)
    clean = cohort.merge(feats, left_on="patientunitstayid", right_index=True, how="left")
    clean["split"] = group_split(clean, "uniquepid", test_size=0.25, stratify="died", random_state=seed)
    xc = base + list(feats.columns)
    model = make_model()
    tr = clean["split"] == "train"
    model.fit(clean.loc[tr, xc], clean.loc[tr, "died"])
    clean["score"] = model.predict_proba(clean[xc])[:, 1]

    # Pipeline with common shortcuts: values after the prediction time, a discharge field
    lfeats, lprov = build_window_features(events, index, aggregations=AGGS, cutoff="none", time_basis="event",
                                          return_provenance=True)
    leaky = cohort.merge(lfeats, left_on="patientunitstayid", right_index=True, how="left")
    leaky["split"] = clean["split"].values
    xl = base + ["unitdischargeoffset"] + list(lfeats.columns)
    lmodel = make_model()
    lmodel.fit(leaky.loc[tr, xl], leaky.loc[tr, "died"])
    leaky["score"] = lmodel.predict_proba(leaky[xl])[:, 1]

    common = dict(label="died", split="split", dataset="eicu", index_time="index_offset", outcome_time="death_offset",
                  eligibility_time="eligible_offset", calendar_time="hospitaldischargeyear", scores="score",
                  split_strategy="patient")
    r_clean = LeakageAudit(clean, feature_columns=xc, events=prov, preprocessor=model, preprocessor_columns=xc, **common).run()
    r_leaky = LeakageAudit(leaky, feature_columns=xl, events=lprov, preprocessor=lmodel, preprocessor_columns=xl, **common).run()
    r_clean.to_html(out / "eicu_clean_report.html")
    r_leaky.to_html(out / "eicu_leaky_report.html")
    r_clean.to_json(out / "eicu_clean_report.json")
    r_leaky.to_json(out / "eicu_leaky_report.json")

    te = clean["split"] == "test"
    perf = pd.DataFrame({
        "clean": evaluate(clean.loc[te, "died"], clean.loc[te, "score"]),
        "leaky": evaluate(leaky.loc[te, "died"], leaky.loc[te, "score"]),
    })
    perf.to_csv(out / "eicu_performance.csv")
    impact = leakage_impact(clean.loc[te, "died"].to_numpy(), clean.loc[te, "score"].to_numpy(), leaky.loc[te, "score"].to_numpy(),
                            n_boot=n_boot, groups=clean.loc[te, "uniquepid"].to_numpy())
    impact.to_csv(out / "eicu_inflation.csv", index=False)
    return {"clean": r_clean, "leaky": r_leaky, "performance": perf, "impact": impact, "n_stays": len(cohort)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="eicu_audit")
    ap.add_argument("--no-vital-periodic", action="store_true", help="Skip the large vitalPeriodic table.")
    ap.add_argument("--max-stays", type=int, help="Random subset of stays for a quick run.")
    args = ap.parse_args()
    res = run(args.root, args.out, use_vital_periodic=not args.no_vital_periodic, max_stays=args.max_stays)
    print(f"Stays: {res['n_stays']}")
    print(res["leaky"].summary())
    print(res["performance"].round(3))
    print(res["impact"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
