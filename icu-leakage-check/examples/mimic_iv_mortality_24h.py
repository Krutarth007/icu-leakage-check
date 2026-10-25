"""Audit a 24-hour in-hospital mortality model built on MIMIC-IV.

Builds the same model twice, once leakage-safe and once with common
shortcuts (whole-stay values, ICU length of stay, same-admission billing
codes), audits both and writes HTML reports plus an inflation table. Only
aggregate results are written to disk.

Usage
-----
    python mimic_iv_mortality_24h.py --root "C:/path/to/mimic-iv-2.2" --out mimic_audit

chartevents and labevents are very large. Convert them to Parquet first
(``icu-leakage-check convert --dataset mimic-iv --tables chartevents,labevents ...``)
and pass the Parquet folder as ``--root``.
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
    attach_entity,
    build_window_features,
    evaluate,
    events_from_table,
    flag_proxy_codes,
    group_split,
    leakage_impact,
    query_table,
    read_table,
)

HOURS = 24
LOOKAHEAD_HOURS = 48  # events loaded beyond the prediction time, used only by the leaky variant
LAB_ITEMS = {
    50813: "lactate", 50912: "creatinine", 51006: "bun", 51301: "wbc", 51265: "platelets",
    50971: "potassium", 50983: "sodium", 50882: "bicarbonate", 50885: "bilirubin", 51222: "hemoglobin",
}
CHART_ITEMS = {
    220045: "heart_rate", 220052: "map_arterial", 220181: "map_nibp", 220210: "resp_rate",
    220277: "spo2", 223762: "temp_c",
}
AGGS = ("last", "min", "max", "mean")


def build_cohort(root) -> pd.DataFrame:
    icu = read_table(root, "icustays", "mimic-iv")
    adm = read_table(root, "admissions", "mimic-iv", columns=[
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "hospital_expire_flag",
    ])
    pat = read_table(root, "patients", "mimic-iv", columns=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"])
    c = icu.merge(adm, on=["subject_id", "hadm_id"]).merge(pat, on="subject_id")
    c = c.sort_values("intime").drop_duplicates("subject_id", keep="first")  # first ICU stay per patient
    c["age_years"] = c["anchor_age"] + (c["intime"].dt.year - c["anchor_year"])
    c["index_time"] = c["intime"] + pd.Timedelta(hours=HOURS)
    c = c[(c["age_years"] >= 18) & (c["outtime"] >= c["index_time"])].copy()  # still in the ICU at the prediction time
    c["died"] = c["hospital_expire_flag"].astype(int)
    c["death_time"] = c["deathtime"].where(c["died"] == 1, pd.NaT)
    c.loc[(c["died"] == 1) & c["death_time"].isna(), "death_time"] = c["dischtime"]
    c = c[~((c["died"] == 1) & (c["death_time"] <= c["index_time"]))]  # alive at the prediction time
    c["eligible_time"] = c["index_time"]
    c["male"] = (c["gender"] == "M").astype(float)
    return c.reset_index(drop=True)


def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


def _windowed(root, table, key, cols, cohort_keys: pd.DataFrame, items):
    """Rows for the cohort between ``lo`` and ``hi`` of each key, filtered while reading."""
    if _has_duckdb():
        sel = ", ".join(f"t.{c}" for c in cols)
        sql = (
            f"SELECT {sel} FROM {{table}} t JOIN coh c ON t.{key} = c.{key} "
            "WHERE t.itemid IN (SELECT v FROM items) AND t.charttime >= c.lo AND t.charttime <= c.hi"
        )
        df = query_table(root, table, sql, dataset="mimic-iv", params={"coh": cohort_keys, "items": list(items)})
        for c in ("charttime", "storetime"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    lo = dict(zip(cohort_keys[key], cohort_keys["lo"]))
    hi = dict(zip(cohort_keys[key], cohort_keys["hi"]))

    def keep(d):
        t = pd.to_datetime(d["charttime"], errors="coerce")
        return d[(t >= d[key].map(lo)) & (t <= d[key].map(hi))]

    return read_table(root, table, "mimic-iv", columns=cols, engine="pandas",
                      filters={key: cohort_keys[key].tolist(), "itemid": list(items)}, row_filter=keep)


def load_events(root, cohort) -> pd.DataFrame:
    stays = cohort[["hadm_id", "stay_id", "admittime", "outtime"]]
    lab_keys = pd.DataFrame({"hadm_id": cohort["hadm_id"], "lo": cohort["admittime"], "hi": cohort["intime"] + pd.Timedelta(hours=LOOKAHEAD_HOURS)})
    lab = _windowed(root, "labevents", "hadm_id", ["hadm_id", "itemid", "charttime", "storetime", "valuenum"], lab_keys, LAB_ITEMS)
    lab_ev = events_from_table(lab, "labevents", "mimic-iv", entity_column="hadm_id")
    lab_ev = attach_entity(lab_ev, stays, "hadm_id", "stay_id", "admittime", "outtime")
    lab_ev["variable"] = pd.to_numeric(lab_ev["variable"]).map(LAB_ITEMS)
    ch_keys = pd.DataFrame({"stay_id": cohort["stay_id"], "lo": cohort["intime"], "hi": cohort["intime"] + pd.Timedelta(hours=LOOKAHEAD_HOURS)})
    ch = _windowed(root, "chartevents", "stay_id", ["stay_id", "itemid", "charttime", "storetime", "valuenum"], ch_keys, CHART_ITEMS)
    ch_ev = events_from_table(ch, "chartevents", "mimic-iv")
    ch_ev["variable"] = pd.to_numeric(ch_ev["variable"]).map(CHART_ITEMS)
    ev = pd.concat([lab_ev, ch_ev], ignore_index=True)
    ev["value"] = pd.to_numeric(ev["value"], errors="coerce")
    return ev.dropna(subset=["value"])


def same_admission_codes(root, cohort) -> pd.DataFrame:
    dx = read_table(root, "diagnoses_icd", "mimic-iv", columns=["hadm_id", "icd_code"], filters={"hadm_id": cohort["hadm_id"].tolist()})
    flags = flag_proxy_codes(dx["icd_code"].unique())
    proxy_codes = set(flags["raw"])
    out = dx.groupby("hadm_id").agg(dx_count=("icd_code", "size"), dx_proxy_code=("icd_code", lambda s: int(s.isin(proxy_codes).any())))
    return out


def make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000)),
    ])


def run(root, out_dir="mimic_audit", max_stays=None, seed=42, n_boot=500):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cohort = build_cohort(root)
    if max_stays:
        cohort = cohort.sample(min(max_stays, len(cohort)), random_state=seed).reset_index(drop=True)
    events = load_events(root, cohort)
    index = cohort[["stay_id", "index_time"]].rename(columns={"stay_id": "entity_id"})
    base = ["age_years", "male"]

    feats, prov = build_window_features(events, index, window=HOURS, aggregations=AGGS, return_provenance=True)
    clean = cohort.merge(feats, left_on="stay_id", right_index=True, how="left")
    clean["split"] = group_split(clean, "subject_id", test_size=0.25, stratify="died", random_state=seed)
    xc = base + list(feats.columns)
    model = make_model()
    tr = clean["split"] == "train"
    model.fit(clean.loc[tr, xc], clean.loc[tr, "died"])
    clean["score"] = model.predict_proba(clean[xc])[:, 1]

    lfeats, lprov = build_window_features(events, index, aggregations=AGGS, cutoff="none", time_basis="event", return_provenance=True)
    leaky = cohort.merge(lfeats, left_on="stay_id", right_index=True, how="left")
    leaky = leaky.merge(same_admission_codes(root, cohort), left_on="hadm_id", right_index=True, how="left")
    leaky["split"] = clean["split"].values
    xl = base + ["los", "dx_count", "dx_proxy_code"] + list(lfeats.columns)
    lmodel = make_model()
    lmodel.fit(leaky.loc[tr, xl], leaky.loc[tr, "died"])
    leaky["score"] = lmodel.predict_proba(leaky[xl])[:, 1]

    common = dict(label="died", split="split", dataset="mimic-iv", index_time="index_time", outcome_time="death_time",
                  eligibility_time="eligible_time", calendar_time="anchor_year_group", scores="score", split_strategy="patient",
                  stay_id="stay_id", encounter_id="hadm_id", patient_id="subject_id")
    r_clean = LeakageAudit(clean, feature_columns=xc, events=prov, preprocessor=model, preprocessor_columns=xc, **common).run()
    sources = {"dx_count": "diagnoses_icd", "dx_proxy_code": "diagnoses_icd.icd_code", "los": "icustays.los"}
    r_leaky = LeakageAudit(leaky, feature_columns=xl, events=lprov, feature_sources=sources, preprocessor=lmodel,
                           preprocessor_columns=xl, **common).run()
    r_clean.to_html(out / "mimic_clean_report.html")
    r_leaky.to_html(out / "mimic_leaky_report.html")
    r_clean.to_json(out / "mimic_clean_report.json")
    r_leaky.to_json(out / "mimic_leaky_report.json")

    te = clean["split"] == "test"
    perf = pd.DataFrame({
        "clean": evaluate(clean.loc[te, "died"], clean.loc[te, "score"]),
        "leaky": evaluate(leaky.loc[te, "died"], leaky.loc[te, "score"]),
    })
    perf.to_csv(out / "mimic_performance.csv")
    impact = leakage_impact(clean.loc[te, "died"].to_numpy(), clean.loc[te, "score"].to_numpy(), leaky.loc[te, "score"].to_numpy(),
                            n_boot=n_boot, groups=clean.loc[te, "subject_id"].to_numpy())
    impact.to_csv(out / "mimic_inflation.csv", index=False)
    return {"clean": r_clean, "leaky": r_leaky, "performance": perf, "impact": impact, "n_stays": len(cohort)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="mimic_audit")
    ap.add_argument("--max-stays", type=int, help="Random subset of stays for a quick run.")
    args = ap.parse_args()
    res = run(args.root, args.out, max_stays=args.max_stays)
    print(f"Stays: {res['n_stays']}")
    print(res["leaky"].summary())
    print(res["performance"].round(3))
    print(res["impact"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
